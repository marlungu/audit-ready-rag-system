from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file =".env", 
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str
    aws_secret_access_key: str

    #S3
    s3_bucket_name: str
    s3_prefix: str = "docs/"

    #Database
    postgres_url: str

    #Models
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024  # Titan V2 supports 256, 512, 1024
    embedding_normalize: bool = True  # Unit vectors for cosine similarity
    chat_model_id: str = "us.anthropic.claude-sonnet-4-6"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    mmr_lambda: float = 0.7
    ivfflat_probes: int = 10

    # Confidence thresholds
    high_confidence_threshold: float = 0.70
    low_confidence_threshold: float = 0.50

    # Generation
    max_context_chunks: int = 3
    temperature: float = 0.0

    # Vector store
    collection_name: str = "uscis_policy_documents"

    # Quality gates
    min_chunk_length: int = 100
    max_chunk_length: int = 3000

    # App
    app_name: str = "USCIS Policy RAG"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    @computed_field
    @property
    def s3_uri(self) -> str:
        return f"s3://{self.s3_bucket_name}/{self.s3_prefix}"
    
    @computed_field
    @property
    def psycopg_url(self) -> str:
        return self.postgres_url.replace(
            "postgresql+psycopg://", "postgresql://", 1
        )


settings = Settings()