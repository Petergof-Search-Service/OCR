from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    OCR_FOLDER_ID: str
    OCR_ACCESS_KEY: str
    OCR_SECRET_KEY: str
    OCR_BUCKET_NAME: str


settings = Settings()
