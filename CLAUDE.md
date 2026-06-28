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
- **Ядро** — класс `YandexOCRAsync` в `src/OCR_async.py`. Хендлер только резолвит путь, дёргает
  статусы и зовёт `asyncio.run(ocr.process_pdf(...))`.
- **Бакет монтируется** в ФС функции (READ_WRITE) в `BUCKET_MOUNT_POINT`
  (`/function/storage/<bucket>`). Никакого boto3/S3 API — работа идёт через примонтированную ФС.
  **Не добавляй boto3/botocore** (в прод-requirements они были по ошибке и не использовались).
- **Шрифт `src/DejaVuSans.ttf` обязателен.** `YandexOCRAsync.__init__` грузит его относительным
  путём `TTFont("DejaVuSans.ttf")` для текстового слоя searchable-PDF. При деплое
  (`source-root: ./src`, `include: **`) он попадает в корень функции. Уберёшь файл — функция
  упадёт на инициализации.
- **Колбэк статусов в API.** `_patch_status()` шлёт `PATCH {API_BASE_URL}/files/by-key/status`
  с заголовком `x-service-key` и телом `{system_key, status, error_message}`. Последовательность:
  `ocr_processing` → (`ocr_done` | `failed`). Watchdog ловит зависшие на `ocr_processing`.
- **Конфиг** — `src/config.py`, всё из env с дефолтами. Тюнинг распознавания — `OCR_*`
  (DPI, JPEG-качество, размер чанка/батча, concurrency, strict-memory-режим для дебага).

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
