"""
Configuration management for embeddings system
"""

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding system"""
    
    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    distance_threshold: float = 1.5
    max_message_length: int = 500
    enable_persistence: bool = True
    
    # Tool-specific settings
    tool_search_k: int = 15
    tool_min_semantic_score: float = 0.4
    tool_max_candidates: int = 10
    
    # Message-specific settings
    message_search_k: int = 10
    message_min_similarity: float = 0.6
    message_max_age_days: int = 7
    message_max_context_pairs: int = 3
    
    # Index settings
    tool_index_dir: str = "./embeddings/indexes/tools"
    message_index_dir: str = "./embeddings/indexes/messages"
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Ensure directories exist
        os.makedirs(self.tool_index_dir, exist_ok=True)
        os.makedirs(self.message_index_dir, exist_ok=True)
