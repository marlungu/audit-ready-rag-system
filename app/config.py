from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file =".env", 
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    #AWS
    aws_region: str = "us-est-1"
    aws_access_key_id: str
    aws_secret_access_key: str

    #S3
    s3_bucket_name: str
    s3_prefix: str = "docs/"

    #Database
    postgres_url: str

    #Models
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    chat_model_id: str = "anthropic.claude-3-7-sonnet-20250219-v1:0"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    search_type: str = "mmr"

    # Vector store
    collection_name: str = "uscis_policy_documents"

settings = Settings()