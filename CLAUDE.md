# CLAUDE.md — OCR (ocr-processing)

Заметки для будущих сессий Claude по этому сервису.

## Что это

Облачная функция **Yandex Cloud** `ocr-processing`, запускаемая **bucket-триггером** при загрузке
PDF в Object Storage (фронт кладёт файл под префикс `incoming/`). Функция распознаёт PDF через
**Yandex Vision OCR API** (`recognizeTextAsync`) и складывает три результата в `result/` того же
бакета: `.txt` (плоский текст), `.json` (постранично) и `.pdf` (исходные изображения с невидимым
текстовым слоем поверх — searchable PDF). Контекст и подробности — в [`README.md`](README.md).

## Главные факты

- **Точка входа:** `handler.handler` (`src/handler.py`). Сигнатура `handler(event, context)`;
  `event` — событие Object Storage, ключ берётся из `messages[0].details.object_id`.
- **Ядро** — пакет `src/ocr/` (оркестратор `YandexOCRAsync` в `ocr/pipeline.py`). Хендлер только
  резолвит путь, дёргает статусы и зовёт `asyncio.run(ocr.process_pdf(...))`. Модули пакета:
  `ratelimit` (token-bucket/backoff/`OCRAuthError`), `client` (OCR API: submit+поллинг, **два
  раздельных лимитера**), `render` (рендер страниц в image-PDF, **byte-aware** под лимит 10 МБ),
  `overlay` (текстовый слой + парсинг ответа), `pdf_split`, `pdf_merge`, `outputs` (`ResultWriter`),
  `layout` (пути), `fsutil`, `pipeline` (оркестрация).
- **Бакет монтируется** в ФС функции (READ_WRITE) в `BUCKET_MOUNT_POINT`
  (`/function/storage/<bucket>`). Никакого boto3/S3 API — работа идёт через примонтированную ФС.
  **Не добавляй boto3/botocore** (в прод-requirements они были по ошибке и не использовались).
- **Шрифт `src/DejaVuSans.ttf` обязателен.** `ocr/overlay.register_font()` грузит его относительным
  путём `TTFont("DejaVuSans.ttf")` для текстового слоя searchable-PDF (cwd функции = корень `src`).
  При деплое (`source-root: ./src`, `include: **`) он попадает в корень функции. Уберёшь файл —
  функция упадёт на инициализации.
- **CPU в одном worker-треде.** `render`/`overlay` — синхронный CPU; в `pipeline` они идут через
  `ThreadPoolExecutor(max_workers=1)`, чтобы не блокировать event loop (сеть OCR параллельно), но
  без конкурентного fitz (PyMuPDF **не потокобезопасен** между потоками).
- **Колбэк статусов в API.** `_patch_status()` шлёт `PATCH {API_BASE_URL}/files/by-key/status`
  с заголовком `x-service-key` и телом `{system_key, status, error_message}`. Последовательность:
  `ocr_processing` → (`ocr_done` | `failed`). Watchdog ловит зависшие на `ocr_processing`.
- **Конфиг** — `src/config.py`, всё из env с дефолтами. Тюнинг распознавания — `OCR_*`
  (DPI, JPEG-качество, размер чанка/батча, concurrency, strict-memory-режим для дебага).
- **Логи** — `src/logging_config.py` (см. ниже). `print()` в коде НЕ использовать — только
  `logger` из `get_logger(...)`, иначе строка потеряет ключ файла и не попадёт в трейс.

## Логирование (как искать трейс по имени файла)

`src/logging_config.py` настраивает root-логгер так, что в **каждой** строке есть
`key=<object_id> req=<request_id>`. Ключ и request_id хранятся в `contextvars` и привязываются
в начале `handler()` (`bind_context`), поэтому автоматически подхватываются во всех корутинах и
тасках обработки одного файла — пробрасывать их по сигнатурам не нужно.

Формат строки: `LEVEL key=<имя файла> req=<request_id> <logger>: <сообщение>`.

Чтобы вытащить весь трейс одного запуска в Yandex Cloud Logging — фильтруй по подстроке с именем
файла (или по `req=<request_id>`):

```bash
yc serverless function logs --id <fn-id> --folder-id <folder> --profile <prof> --since 3h \
  | grep 'key=incoming/<имя файла>'
```

Уровень — env `LOG_LEVEL` (по умолчанию `INFO`). Ключевые точки: старт хендлера (bucket/key/
request_id), путь и размер входного файла, конфиг OCR, число страниц/чанков, тайминг по чанкам и
суммарный, статусы в API, ретраи/ошибки OCR API, фейлы батчей (с traceback через
`logger.exception`), пути результатов. Падение пишется `logger.exception` и в хендлере, и в
`process_pdf` (с указанием, сколько чанков успели обработать).

## Прод-баг, который чинит этот репозиторий (важно)

Раньше код жил отдельными копиями в каждом облаке и разошёлся. На **проде**
`OCR_async.py` оказался **побайтовой копией `handler.py`** (без класса `YandexOCRAsync`), из-за
чего `from OCR_async import YandexOCRAsync` падал с `ImportError` и OCR на проде не работал.
Теперь источник один — `src/` (взят из рабочей testing-версии), и CI деплоит его в оба облака.
Не возвращай раздельные копии.

## Связь с другими сервисами

```
Frontend → API кладёт PDF в bucket/incoming/ → [ЭТА ФУНКЦИЯ] → bucket/result/{txt,json,pdf}
           статусы: uploaded → ocr_processing → ocr_done → (RAG) rag_indexing → indexed
```

- **Фронт/API** (`../Frontend`, `../API`) — загружают файл и хранят его `status`.
- **Watchdog** (`../Watchdog`) — раз в минуту помечает `dead` файлы, застрявшие в
  `ocr_processing` дольше лимита (например, если YC убил функцию по таймауту/OOM).
- **RAG** (`upload-file-to-rag-base`) — следующий шаг после `ocr_done`.

## Линт

```bash
pip install ruff==0.9.2
ruff check src
```

Юнит-тестов нет: пайплайн IO-тяжёлый (рендер PDF, сетевой OCR). CI-гейт — только ruff по `src`
(см. `pyproject.toml`, `select = ["E","F","I","UP","B"]`, `target py312` для статанализа,
рантайм в облаке — `python314`).

## Деплой

- CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Test и prod — **разные
  облака/каталоги** YC; функция в обоих зовётся `ocr-processing`. PR → деплой в testing-каталог;
  мерж в `main` → деплой в prod-каталог. Рантайм `python314`, `memory 2GB`,
  `execution-timeout 3600s`, `concurrency 1`.
- **Деплой — напрямую через `yc` CLI** ([`.github/deploy.sh`](.github/deploy.sh)), НЕ через
  экшен `yc-actions/yc-sls-function`: тот экшен **не поддерживает вход `mounts`** (нет в списке
  валидных инпутов), а OCR без смонтированного бакета не работает. Скрипт ставит yc CLI,
  логинится по SA-ключу, зипует `src/` и зовёт `yc serverless function version create`.
- ⚠️ **Каждая версия YC-функции — полная спека.** `--mount`, `--network-id`, `--environment`,
  `--service-account-id`, `--concurrency` задаются в КАЖДОМ деплое. Забудешь `--mount` — новая
  версия потеряет монтирование бакета, и функция перестанет видеть входной файл. Это главное
  отличие от watchdog (там нет ни монтирования, ни VPC).
- `environment` выставляется **на уровне версии функции** (как в watchdog), а не из забандленного
  `.env` (в `src/` его нет). `config.py` всё равно вызывает `load_dotenv()` — это безвредно
  (нет файла → no-op) и удобно для локальной разработки; функц-env он не перетирает.
- CI публикует только версии. Сами функции и bucket-триггеры созданы один раз вручную (см. README).

## Rate-limiting OCR API и масштабирование (важно)

OCR API имеет **раздельные** квоты RPS на каталог (Yandex): submit `recognizeTextAsync` ≈10 rps,
поллинг/получение `getRecognition` — по ≈50 rps. Плюс лимиты запроса: **≤200 стр**, **файл ≤10 МБ**.
Защита от 429 — в два слоя:

1. **Per-file (в коде):** **два раздельных** token-bucket — `OCR_SUBMIT_RPS` (дефолт 8, под квоту
   submit) и `OCR_POLL_RPS` (дефолт 40, под квоту поллинга) — + экспоненциальный backoff с
   `Retry-After` (`ocr/client.py`). Ограничивают RPS **одного инстанса** (= одного файла).
   Чтобы кратно сократить число submit-запросов, отправляем **много страниц на запрос**
   (`OCR_BATCH_SIZE`, дефолт 10), а `render` режет батч по `OCR_MAX_BATCH_BYTES` (~9 МБ), чтобы
   запрос не превысил лимит 10 МБ.
2. **Глобально (инфра):** лимитеры НЕ координируют разные инстансы — каждый файл это отдельный
   вызов функции. Поэтому стоит **scaling policy** `zone-instances-limit=2` на теге `$latest`
   (`yc serverless function set-scaling-policy`). Сеть на 3 подсетях/зонах ⇒ одновременно ≤ `3·2=6`
   инстансов ⇒ пик глобального submit-RPS ≈ `6 · OCR_SUBMIT_RPS`.

**После поднятия квоты** (например до 60 rps): достаточно поднять env `OCR_SUBMIT_RPS` (~55) и при
желании `OCR_BATCH_SIZE` — код менять не нужно.

**Тюнинг:** ориентир — лог `Pages summary: total/ok/failed` и число `poll HTTP 429` / `recognize
attempt … HTTP 429`. Много 429/таймаутов → снижай `OCR_SUBMIT_RPS`/`OCR_POLL_RPS` или
`zone-instances-limit`. Мало и медленно → повышай. Регуляторы: `zone-instances-limit`
(параллельность файлов), `OCR_SUBMIT_RPS`/`OCR_POLL_RPS` (нагрузка на файл), `OCR_BATCH_SIZE`
(страниц на запрос).

⚠️ **Кап инстансов может ронять события триггера.** При упоре в лимит часть вызовов от bucket-триггера
не обработается; у прод-триггера `retry_attempts=1`, поэтому файл рискует застрять в `uploaded`
(watchdog потом пометит `dead`). Если поднимаешь параллельную загрузку — увеличь `retry_attempts`
триггера или подними `zone-instances-limit`. Альтернатива жёсткому глобальному лимиту без потери
параллельности — распределённый лимитер (YDB Coordination/Redis), но это +инфра (пока не делаем).

Политика масштабирования — на уровне функции (не версии), переживает деплои:
```bash
yc serverless function set-scaling-policy --id <fn-id> --folder-id <folder> --profile <prof> \
  --tag '$latest' --zone-instances-limit 2
yc serverless function list-scaling-policies --id <fn-id> --folder-id <folder> --profile <prof>
```

## Yandex Cloud (инфраструктура и доступы)

**Два независимых окружения — два разных облака/каталога.** В каждом своя функция, свой триггер,
свой сервисный аккаунт, свой бакет.

| | testing | prod |
|---|---|---|
| folder-id | `b1g45gcej7fc0v27s3fn` | `b1g0gm5epaeepgfov6ni` |
| функция `ocr-processing` | `d4e3u59k6ku61rb6l0dt` | `d4e6ucu8v4rfin255fbt` |
| bucket-триггер | `upload-ocr-file` (`a1sv1tjia6j3eapek3rs`) | `upload-file-ocr-trigger` (`a1splanr3354v1d5k8sv`) |
| бакет (вход+выход) | `petergof-testing-backet` | `petergof-incoming-prod` |
| mount point | `/function/storage/petergof-testing-backet` | `/function/storage/petergof-incoming-prod` |
| SA `petergof-robot` | `ajeccs8j8b7igcjacd0r` | `ajerbgv6jbhvacbev1id` |
| runtime / concurrency | `python314` / 1 | `python314` / 1 |

`FOLDER_ID` (env, x-folder-id для тарификации OCR API) сейчас в обоих окружениях =
`b1g0gm5epaeepgfov6ni`.

### yc CLI профили

- `petergof-robot` — testing (активен по умолчанию).
- `wd-prod` — prod (тот же приём, что в watchdog):
  ```bash
  yc config profile create wd-prod   # один раз
  yc config set service-account-key ~/petergof-robot-key-prod.json
  yc config set folder-id b1g0gm5epaeepgfov6ni
  ```
- Команды для прода — с `--profile wd-prod`. После работы возвращай активным `petergof-robot`
  (`yc config profile activate petergof-robot`).

### Полезные команды

```bash
# Функции / версии / триггеры
yc serverless function list --folder-id <folder> --profile <prof>
yc serverless function version list --function-id <fn-id> --folder-id <folder> --profile <prof>
yc serverless function version get --id <version-id> --folder-id <folder> --profile <prof>
yc serverless trigger get --id <trigger-id> --folder-id <folder> --profile <prof>

# Логи функции (диагностика прогона OCR)
yc serverless function logs --id <fn-id> --folder-id <folder> --profile <prof> --since 1h

# Содержимое бакета (вход/результаты)
yc storage s3api list-objects --bucket <bucket> --prefix incoming/ --profile <prof>
yc storage s3api list-objects --bucket <bucket> --prefix result/  --profile <prof>
```

### Подводные камни (важно)

- В обоих облаках SA называется **одинаково — `petergof-robot`**, но это **разные** SA с разными
  id. У каждого `editor`/доступ только в своём облаке. Этого SA достаточно и для деплоя, и для
  RW-монтирования бакета, и чтобы триггер вызывал функцию.
- `service-account: petergof-robot` в CI резолвится по имени внутри целевого folder — у каждого
  окружения подставится свой SA.
- `mounts` надо передавать на каждом деплое (см. выше).
- `concurrency = 1` (одна страница/инстанс ради памяти). Менялась вручную; экшен её не выставляет —
  если важно, проверяй/правь через `yc`.
- Рантайм `python314` (как у соседних RAG-функций), не `python312`.
- Bucket-триггер ссылается на функцию и SA по **id**; prefix — `incoming`.

### GitHub-секреты/переменные репозитория OCR

- Secrets: `YC_SA_JSON_CREDENTIALS_TEST`/`_PROD` (JSON-ключи роботов),
  `YC_FOLDER_ID_TEST`/`_PROD`, `YANDEX_API_KEY` (ключ Vision OCR API),
  `CLOUD_FUNCTION_API_KEY` (**org-секрет**; visibility должен включать этот репозиторий, иначе
  приедет пустым и YC упадёт с `Illegal value of environment variable`).
- Variables (`vars.`): `API_BASE_URL_TEST`/`_PROD`.
- Бакеты, точки монтирования, SA-id, network-id, `FOLDER_ID` и `INPUT_PREFIX` зашиты прямо в
  `ci.yml`/`deploy.sh` (стабильная инфраструктура).

## Чего не делать

- Не возвращай раздельные prod/test-копии кода — источник один (`src/`).
- Не добавляй boto3/S3 — бакет монтируется как ФС.
- Не ломай относительный путь к `DejaVuSans.ttf` и не удаляй файл из `src/`.
- Не меняй формат `_patch_status` (`system_key`/`status`/`error_message`, заголовок
  `x-service-key`) без синхронизации с `../API` (эндпоинт `PATCH /files/by-key/status`).
- Не убирай `mounts`/`environment` из джоб деплоя.
