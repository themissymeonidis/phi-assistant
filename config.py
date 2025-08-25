import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "local_assistant"
    user: str = "postgres"
    password: Optional[str] = None
    
    def __post_init__(self):
        self.host = os.getenv("PG_HOST", self.host)
        self.database = os.getenv("PG_DBNAME", self.database)
        self.user = os.getenv("PG_USER", self.user)
        self.password = os.getenv("PG_PASSWORD", self.password)

@dataclass
class ModelConfig:
    model_path: str = "./models/Phi-3-mini-4k-instruct-q4.gguf"
    context_size: int = 4096
    threads: int = 8
    gpu_layers: int = 35
    max_tokens: int = 512
    temperature: float = 0.7

@dataclass
class FaissConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    distance_threshold: float = 1.5
    top_k: int = 3

@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    log_level: str = "INFO"
    
    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

# Global config instance
config = Config()
