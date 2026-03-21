import json
import boto3
import asyncio
from config import S3_ACCESS_KEY, S3_SECRET_KEY, FOLDER_ID, YANDEX_API_KEY


def handler(event, context):
    print("Handler started")
    print(json.dumps(event, ensure_ascii=False))

    message = event["messages"][0]
    details = message["details"]

    bucket = details["bucket_id"]
    key = details["object_id"]

    if not key.startswith("OCR-request/"):
        print("Skip non-input object:", key)
        return {"statusCode": 200}

    s3 = boto3.client(
        "s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

    print("Downloading object into memory...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    input_pdf_bytes = obj["Body"].read()

    from OCR_async import YandexOCRAsync

    ocr = YandexOCRAsync(
        api_key=YANDEX_API_KEY,
        folder_id=FOLDER_ID,
        bucket=bucket,
        key=key,
        s3_client=s3
    )

    asyncio.run(
        ocr.process_pdf(
            input_pdf_bytes=input_pdf_bytes,
            max_concurrent=10,
            cleanup_tmp_s3=True,  # True, если хочешь удалять OCR-tmp после завершения
        )
    )

    return {"statusCode": 200}
