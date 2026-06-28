#!/usr/bin/env bash
# Деплой новой версии функции ocr-processing в Yandex Cloud.
#
# Экшен yc-actions/yc-sls-function НЕ умеет монтировать бакет, а OCR без
# смонтированного бакета не работает, поэтому деплоим напрямую через yc CLI.
# Каждая версия YC-функции — ПОЛНАЯ спека: mount, network, env, SA задаём
# целиком на каждом деплое, иначе новая версия потеряет эти настройки.
#
# Параметры приходят из env шага workflow:
#   YC_SA_JSON, FOLDER_ID, SA_ID, NETWORK_ID, BUCKET,
#   OCR_API_KEY, API_BASE_URL, CLOUD_FUNCTION_API_KEY
# и из workflow-level env: OCR_API_FOLDER_ID, RUNTIME, MEMORY,
#   EXECUTION_TIMEOUT, CONCURRENCY.
set -euo pipefail

# 1. Установка yc CLI
curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash -s -- -n -i "$HOME/yandex-cloud"
export PATH="$HOME/yandex-cloud/bin:$PATH"

# 2. Аутентификация по ключу сервисного аккаунта
printf '%s' "$YC_SA_JSON" > "$HOME/sa-key.json"
trap 'rm -f "$HOME/sa-key.json"' EXIT
yc config profile create deploy >/dev/null 2>&1 || yc config profile activate deploy
yc config set service-account-key "$HOME/sa-key.json"
yc config set folder-id "$FOLDER_ID"

# 3. Упаковка исходников (handler.py и DejaVuSans.ttf — в корне архива)
( cd src && zip -qr "$GITHUB_WORKSPACE/function.zip" . )

# 4. Создание новой версии с полной спекой
yc serverless function version create \
  --function-name ocr-processing \
  --runtime "$RUNTIME" \
  --entrypoint handler.handler \
  --memory "$MEMORY" \
  --execution-timeout "$EXECUTION_TIMEOUT" \
  --concurrency "$CONCURRENCY" \
  --service-account-id "$SA_ID" \
  --network-id "$NETWORK_ID" \
  --source-path "$GITHUB_WORKSPACE/function.zip" \
  --mount "type=object-storage,mount-point=$BUCKET,bucket=$BUCKET,mode=rw" \
  --environment "FOLDER_ID=$OCR_API_FOLDER_ID" \
  --environment "YANDEX_API_KEY=$OCR_API_KEY" \
  --environment "BUCKET_MOUNT_POINT=/function/storage/$BUCKET" \
  --environment "INPUT_PREFIX=incoming/" \
  --environment "API_BASE_URL=$API_BASE_URL" \
  --environment "CLOUD_FUNCTION_API_KEY=$CLOUD_FUNCTION_API_KEY"
