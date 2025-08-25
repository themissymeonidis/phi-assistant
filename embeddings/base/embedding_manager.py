"""
Base embedding manager with common functionality
"""

from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from embeddings.base.faiss_persistence import FaissPersistenceManager
from embeddings.config import EmbeddingConfig
from typing import Dict, List, Optional, Any
import numpy as np
import faiss


class BaseEmbeddingManager(ABC):
    """Base class for embedding managers with common functionality"""
    
    def __init__(self, embedding_model_name: str, index_dir: str, config: Optional[EmbeddingConfig] = None):
        """
        Initialize base embedding manager
        
        Args:
            embedding_model_name: Name of the embedding model to use
            index_dir: Directory for storing FAISS indexes
            config: Configuration object
        """
        self.config = config or EmbeddingConfig()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.persistence_manager = FaissPersistenceManager(index_dir)
        self.index: Optional[faiss.Index] = None
        self.mapping: Dict[int, Dict] = {}
    
    @abstractmethod
    def _build_index(self, vectors: np.ndarray, mapping: Dict[int, Dict]) -> faiss.Index:
        """Build FAISS index from vectors and mapping"""
        pass
    
    @abstractmethod
    def _search_index(self, query_vector: np.ndarray, k: int) -> tuple:
        """Search index with query vector"""
        pass
    
    def _create_empty_index(self, dimension: int) -> faiss.Index:
        """Create an empty FAISS index with given dimension"""
        return faiss.IndexFlatL2(dimension)
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query string to vector"""
        return self.embedding_model.encode([query]).astype('float32')
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts to vectors"""
        return self.embedding_model.encode(texts).astype('float32')
