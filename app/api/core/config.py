from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field(default="groq", env="LLM_PROVIDER")
    llm_model: str = Field(default="llama-3.1-8b-instant", env="LLM_MODEL")
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    hf_token: str = Field(default="", env="HF_TOKEN")

    # Vector DB
    vector_db: str = Field(default="chroma", env="VECTOR_DB")
    vector_db_path: str = Field(
        default="./data/processed/vectorstore",
        env="VECTOR_DB_PATH"
    )

    # Chunking
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(default=5, env="TOP_K")
    hybrid_alpha: float = Field(default=0.7, env="HYBRID_ALPHA")

    # App
    app_env: str = Field(default="development", env="APP_ENV")
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()