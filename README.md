# OCR (ocr-processing)

**Облачная функция Yandex Cloud**, которая распознаёт загруженные PDF и превращает их в текст.
Запускается **bucket-триггером**: как только файл попадает в Object Storage под префикс
`incoming/`, функция распознаёт его через **Yandex Vision OCR API** и кладёт результаты в
`result/`.

## Зачем нужен

В системе Petergof RAG загруженный документ проходит несколько стадий, которыми управляют
облачные функции:

```
загрузка → S3/incoming → функция OCR → функция RAG
        uploaded → ocr_processing → ocr_done → rag_indexing → indexed
```

OCR — второй шаг. Он берёт «сырой» PDF, распознаёт его и отдаёт дальше три формата результата,
а также сообщает API о смене статуса (`ocr_processing` → `ocr_done`/`failed`), чтобы фронтенд
показывал актуальное состояние, а [Watchdog](../Watchdog) мог поймать зависшую обработку.

## Как работает

1. Фронт/API загружают PDF в бакет под `incoming/`. Срабатывает bucket-триггер и вызывает функцию.
2. `handler.handler` берёт ключ объекта из события (`messages[0].details.object_id`), отсеивает
   объекты не из `incoming/`, резолвит путь в **смонтированном** бакете и шлёт API статус
   `ocr_processing`.
3. `YandexOCRAsync.process_pdf()` (`src/OCR_async.py`):
   - режет PDF на **чанки** и **батчи**, рендерит страницы в изображения (PyMuPDF) заданного
     DPI/качества;
   - отправляет их в Yandex Vision OCR (`recognizeTextAsync`) и опрашивает результат;
   - собирает текст и координаты блоков, строит по ним невидимый текстовый слой (reportlab +
     `DejaVuSans.ttf`) поверх изображения страницы.
4. Итог пишется в `result/` того же бакета:
   - `result/txt-files/<name>.txt` — плоский текст;
   - `result/json-files/<name>.json` — постранично (`{"data":[{"page","text"}, ...]}`);
   - `result/pdf-files/<name>.pdf` — searchable PDF (картинки + невидимый текст).
5. Временные файлы пишутся в `tmp/<name>/` и удаляются по завершении. По итогу — статус `ocr_done`
   (или `failed` с текстом ошибки).

Бакет **примонтирован** в файловую систему функции (`BUCKET_MOUNT_POINT`,
`/function/storage/<bucket>`, режим READ_WRITE) — отдельного S3-клиента нет, всё через ФС.

## Структура

```
src/handler.py        # точка входа handler.handler + резолв пути + колбэк статусов
src/OCR_async.py      # класс YandexOCRAsync — весь пайплайн распознавания
src/config.py         # чтение env + параметры OCR (с дефолтами)
src/logging_config.py # логирование с ключом файла/request_id в каждой строке
src/requirements.txt  # рантайм-зависимости, которые ставит Yandex Cloud
src/DejaVuSans.ttf    # шрифт для текстового слоя (обязателен в рантайме)
.github/workflows/    # CI: ruff + деплой в оба облака
.github/deploy.sh     # деплой версии функции через yc CLI
```

## Конфигурация

Все параметры читаются из переменных окружения (`src/config.py`). См. [`.env.example`](.env.example).

| Переменная               | Описание                                                            |
|--------------------------|---------------------------------------------------------------------|
| `FOLDER_ID`              | Каталог YC для тарификации Vision OCR API (`x-folder-id`)            |
| `YANDEX_API_KEY`         | API-ключ SA с ролью `ai.vision` (`Authorization: Api-Key ...`)       |
| `BUCKET_MOUNT_POINT`     | Точка монтирования бакета (`/function/storage/<bucket>`)             |
| `INPUT_PREFIX`           | Префикс входных файлов (`incoming/`, совпадает с prefix триггера)    |
| `TMP_PREFIX`             | Префикс временных файлов в бакете (`tmp`)                            |
| `RESULT_PREFIX`          | Префикс результатов в бакете (`result`)                             |
| `OCR_STRICT_MEMORY_MODE` | `1` — строго по одной странице (дебаг/память), `0` — параллельно     |
| `OCR_MAX_CONCURRENT`     | Параллелизм запросов/соединений                                     |
| `OCR_BATCH_SIZE`         | Страниц в батче распознавания                                      |
| `OCR_CHUNK_SIZE`         | Страниц в чанке                                                    |
| `OCR_DPI`                | DPI рендера страниц                                                 |
| `OCR_JPEG_QUALITY`       | Качество JPEG                                                       |
| `API_BASE_URL`           | Базовый URL API с префиксом версии, без слеша в конце               |
| `CLOUD_FUNCTION_API_KEY` | Сервисный ключ; должен совпадать с `CLOUD_FUNCTION_API_KEY` в API    |

## Логи

Все логи пишутся через `logging` (`src/logging_config.py`) и в **каждой** строке содержат
`key=<имя файла> req=<request_id>`. Чтобы посмотреть трейс конкретного запуска в Yandex Cloud
Logging, фильтруй по имени файла:

```bash
yc serverless function logs --id <fn-id> --folder-id <folder> --profile <prof> --since 3h \
  | grep 'key=incoming/<имя файла>'
```

Уровень задаётся переменной `LOG_LEVEL` (по умолчанию `INFO`). Логируются: старт обработки,
путь и размер файла, число страниц/чанков, тайминги, статусы, ретраи и ошибки OCR API, падения
батчей с traceback.

## Локальная разработка

```bash
python -m venv .venv && source .venv/bin/activate
pip install ruff==0.9.2
ruff check src
```

Юнит-тестов нет — пайплайн IO-тяжёлый (рендер PDF + сетевой OCR), поэтому CI-гейт ограничен
линтером. Запуск пайплайна целиком имеет смысл только в облаке со смонтированным бакетом.

## Деплой (CI/CD)

Test и prod живут в **разных облаках/каталогах** Yandex Cloud. Функция в обоих называется
одинаково — `ocr-processing`; различаются креды, `folder-id`, бакет и `API_BASE_URL`.
[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **Любой push и PR** → джоба `lint` (ruff по `src`).
- **Pull request** → деплой новой версии `ocr-processing` в **testing**-каталог.
- **Мерж в `main`** → деплой новой версии `ocr-processing` в **prod**-каталог.

Деплой выполняется **напрямую через `yc` CLI** ([`.github/deploy.sh`](.github/deploy.sh)):
рантайм `python314`, `memory 2GB`, `execution-timeout 3600s`, `concurrency 1`,
`service-account-id` своего окружения. Готовый экшен
[`yc-actions/yc-sls-function`](https://github.com/yc-actions/yc-sls-function) **не используется**,
потому что не умеет монтировать бакет (`mounts`), а без него OCR не видит входной файл.

> ⚠️ Каждая версия YC-функции — **полная спека**. Монтирование бакета (`--mount`), VPC
> (`--network-id`), `--environment` и `--service-account-id` задаются в каждом деплое; иначе новая
> версия потеряет эти настройки. Переменные окружения выставляются на уровне версии функции (как
> в watchdog), а не из `.env` в коде.

### Окружения

| Окружение | Когда деплоится     | Креды/переменные в GitHub                                                   |
|-----------|---------------------|----------------------------------------------------------------------------|
| testing   | при открытии PR     | `YC_SA_JSON_CREDENTIALS_TEST`, `YC_FOLDER_ID_TEST`, `vars.API_BASE_URL_TEST` |
| prod      | при мерже в `main`  | `YC_SA_JSON_CREDENTIALS_PROD`, `YC_FOLDER_ID_PROD`, `vars.API_BASE_URL_PROD` |

### Требуемые настройки GitHub

Secrets:

| Секрет                          | Описание                                                          |
|---------------------------------|------------------------------------------------------------------|
| `YC_SA_JSON_CREDENTIALS_TEST`   | JSON-ключ SA `petergof-robot` для testing-облака (`editor`)       |
| `YC_SA_JSON_CREDENTIALS_PROD`   | JSON-ключ SA `petergof-robot` для prod-облака                     |
| `YC_FOLDER_ID_TEST`             | ID каталога testing (`b1g45gcej7fc0v27s3fn`)                      |
| `YC_FOLDER_ID_PROD`             | ID каталога prod (`b1g0gm5epaeepgfov6ni`)                         |
| `YANDEX_API_KEY`                | API-ключ Vision OCR                                              |
| `CLOUD_FUNCTION_API_KEY`        | Общий сервисный ключ (то же значение, что ожидает API)           |

Variables (`vars.`):

| Переменная           | Описание             |
|----------------------|----------------------|
| `API_BASE_URL_TEST`  | Базовый URL тест-API |
| `API_BASE_URL_PROD`  | Базовый URL прод-API |

Бакеты, точки монтирования, `FOLDER_ID` и `INPUT_PREFIX` зашиты в `ci.yml` (стабильная инфра).

## Разовая настройка в Yandex Cloud

CI публикует только **версии**. Сама функция и её bucket-триггер созданы один раз **в каждом
облаке** через CLI `yc` (SA `petergof-robot` с ролью `editor`):

```bash
# 1. Функция (CI-экшен также создаёт её при первом запуске, если её нет)
yc serverless function create --name ocr-processing --folder-id <folder-id>

# 2. Bucket-триггер: запуск при создании объекта под префиксом incoming/
yc serverless trigger create object-storage \
  --name upload-ocr-file \
  --bucket-id <bucket> \
  --prefix incoming \
  --events 'create-object' \
  --batch-size 1 \
  --invoke-function-name ocr-processing \
  --invoke-function-service-account-name petergof-robot \
  --folder-id <folder-id>

# 3. JSON-ключ SA для секрета YC_SA_JSON_CREDENTIALS_{TEST,PROD}
yc iam key create --service-account-name petergof-robot --output key.json
```

> После вставки `key.json` в GitHub-секрет удалите файл (`rm key.json`) — это приватный ключ.

Точные id функций/триггеров/бакетов и подводные камни по облаку — в [`CLAUDE.md`](CLAUDE.md).

## Контракт с API

OCR зависит от одного эндпоинта API с авторизацией по сервисному ключу (заголовок `x-service-key`):

- `PATCH /files/by-key/status` → тело `{ "system_key", "status", "error_message" }`.
  Шлётся со статусами `ocr_processing`, `ocr_done`, `failed`.

## Границы ответственности

Функция только распознаёт и пишет результаты в бакет + сообщает статусы. Она не индексирует
(это RAG-функция) и не воскрешает упавшие задачи (это [Watchdog](../Watchdog)). Наблюдаемость —
через логи функции в Yandex Cloud.
