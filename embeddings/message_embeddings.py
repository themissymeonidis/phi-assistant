"""
Message Embedding Manager for Semantic Conversation History Search
Embeds and indexes conversation messages for contextual retrieval using FAISS
"""

import numpy as np
import faiss
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer

from utils.database import db_manager
from utils.conversation_logger import conversation_logger
from embeddings.faiss_persistence import FaissPersistenceManager


class MessageEmbeddingManager:
    """
    Manages FAISS-based semantic search for conversation messages
    """
    
    def __init__(self, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 index_dir: str = "./embeddings/message_indexes",
                 max_message_length: int = 500,
                 enable_persistence: bool = True):
        """
        Initialize the message embedding manager
        
        Args:
            embedding_model_name: Name of the embedding model to use
            index_dir: Directory for storing FAISS indexes
            max_message_length: Maximum message length to embed (longer messages get truncated)
            enable_persistence: Whether to enable index persistence
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.max_message_length = max_message_length
        self.enable_persistence = enable_persistence
        
        # Index management
        self.index: Optional[faiss.Index] = None
        self.message_mapping: Dict[int, Dict] = {}  # Maps FAISS index position to message data
        self.last_indexed_message_id = 0
        self.index_build_time = None
        
        # Persistence
        if enable_persistence:
            self.persistence_manager = FaissPersistenceManager(index_dir)
            self.persistence_manager.index_file = self.persistence_manager.index_dir / "messages.faiss"
            self.persistence_manager.metadata_file = self.persistence_manager.index_dir / "messages_metadata.json"
            self.persistence_manager.mapping_file = self.persistence_manager.index_dir / "messages_mapping.json"
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load the FAISS index for messages"""
        try:
            if self.enable_persistence:
                # Try to load existing index
                loaded_index, loaded_mapping, is_valid = self._load_persisted_index()
                
                if is_valid and loaded_index and loaded_mapping:
                    self.index = loaded_index
                    self.message_mapping = {int(k): v for k, v in loaded_mapping.items()}
                    self.last_indexed_message_id = max([msg['id'] for msg in self.message_mapping.values()]) if self.message_mapping else 0
                    
                    conversation_logger.log_system_event(
                        "message_index_loaded",
                        f"Loaded message index with {len(self.message_mapping)} messages"
                    )
                    
                    # Check for new messages and update index
                    self._update_index_incrementally()
                    return
            
            # Build index from scratch
            self._build_index_from_scratch()
            
        except Exception as e:
            conversation_logger.log_error("message_index_init_failed", str(e), "Failed to initialize message index")
            # Create empty index as fallback
            self._create_empty_index()
    
    def _load_persisted_index(self) -> Tuple[Optional[faiss.Index], Optional[Dict], bool]:
        """Load persisted index if available and valid"""
        try:
            # Check if index files exist
            if not all(f.exists() for f in [
                self.persistence_manager.index_file,
                self.persistence_manager.metadata_file,
                self.persistence_manager.mapping_file
            ]):
                return None, None, False
            
            # Load metadata
            with open(self.persistence_manager.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate metadata
            if metadata.get("embedding_model") != self.embedding_model_name:
                conversation_logger.log_system_event(
                    "message_index_model_mismatch",
                    f"Index model {metadata.get('embedding_model')} != current {self.embedding_model_name}"
                )
                return None, None, False
            
            # Load index and mapping
            index = faiss.read_index(str(self.persistence_manager.index_file))
            
            with open(self.persistence_manager.mapping_file, 'r') as f:
                mapping = json.load(f)
            
            return index, mapping, True
            
        except Exception as e:
            conversation_logger.log_error("message_index_load_failed", str(e), "Failed to load persisted message index")
            return None, None, False
    
    def _build_index_from_scratch(self):
        """Build FAISS index from all messages in database"""
        try:
            conversation_logger.log_system_event("message_index_build_start", "Building message index from scratch")
            start_time = time.time()
            
            # Get all messages suitable for embedding
            messages = self._fetch_messages_for_embedding()
            
            if not messages:
                self._create_empty_index()
                return
            
            # Generate embeddings
            embeddings = self._generate_embeddings_for_messages(messages)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # Create mapping
            self.message_mapping = {i: msg for i, msg in enumerate(messages)}
            self.last_indexed_message_id = max([msg['id'] for msg in messages])
            self.index_build_time = time.time() - start_time
            
            # Save to disk if persistence enabled
            if self.enable_persistence:
                self._save_index_to_disk(messages)
            
            conversation_logger.log_system_event(
                "message_index_build_complete",
                f"Built message index: {len(messages)} messages, {self.index_build_time:.2f}s"
            )
            
        except Exception as e:
            conversation_logger.log_error("message_index_build_failed", str(e), "Failed to build message index")
            self._create_empty_index()
    
    def _fetch_messages_for_embedding(self, since_message_id: int = 0) -> List[Dict]:
        """
        Fetch messages from database suitable for embedding
        
        Args:
            since_message_id: Only fetch messages with ID > this value
            
        Returns:
            List of message dictionaries
        """
        try:
            # Only embed user and assistant messages (not system/tool messages)
            # Also filter out very short messages that don't provide meaningful context
            query = """
                SELECT m.id, m.conversation_id, m.role, m.content, m.sequence_number, 
                       m.created_at, c.title as conversation_title, m.tool_name, m.tool_id
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.id > %s
                AND m.role IN ('user', 'assistant')
                AND LENGTH(m.content) >= 10
                AND m.content NOT LIKE 'exit'
                AND m.content NOT LIKE 'help'
                AND m.content NOT LIKE 'clear'
                ORDER BY m.id ASC
            """
            
            result = db_manager.execute_query(query, (since_message_id,))
            
            messages = []
            for row in result:
                # Truncate long messages for embedding
                content = row[3]
                if len(content) > self.max_message_length:
                    content = content[:self.max_message_length] + "..."
                
                message_data = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'role': row[2],
                    'content': content,
                    'original_content': row[3],  # Keep original for retrieval
                    'sequence_number': row[4],
                    'created_at': row[5].isoformat() if row[5] else None,
                    'conversation_title': row[6] or 'Untitled',
                    'tool_name': row[7],  # NEW: Include tool name
                    'tool_id': row[8]     # NEW: Include tool ID
                }
                messages.append(message_data)
            
            return messages
            
        except Exception as e:
            conversation_logger.log_error("message_fetch_failed", str(e), "Failed to fetch messages for embedding")
            return []
    
    def _generate_embeddings_for_messages(self, messages: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for a list of messages
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            NumPy array of embeddings
        """
        try:
            # Create enhanced text for embedding that includes context
            texts_to_embed = []
            
            for msg in messages:
                # Create enhanced text with role and conversation context
                enhanced_text = f"{msg['role']}: {msg['content']}"
                
                # Add conversation title for additional context
                if msg['conversation_title'] and msg['conversation_title'] != 'Untitled':
                    enhanced_text = f"[{msg['conversation_title']}] {enhanced_text}"
                
                texts_to_embed.append(enhanced_text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts_to_embed, show_progress_bar=False)
            return np.array(embeddings, dtype='float32')
            
        except Exception as e:
            conversation_logger.log_error("embedding_generation_failed", str(e), "Failed to generate message embeddings")
            raise
    
    def _create_empty_index(self):
        """Create an empty FAISS index"""
        # Create with standard embedding dimension
        dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.message_mapping = {}
        self.last_indexed_message_id = 0
        
        conversation_logger.log_system_event("empty_message_index_created", "Created empty message index")
    
    def _update_index_incrementally(self):
        """Update index with new messages since last build"""
        try:
            # Fetch new messages
            new_messages = self._fetch_messages_for_embedding(self.last_indexed_message_id)
            
            if not new_messages:
                return
            
            # Generate embeddings for new messages
            new_embeddings = self._generate_embeddings_for_messages(new_messages)
            
            # Add to existing index
            start_idx = len(self.message_mapping)
            self.index.add(new_embeddings)
            
            # Update mapping
            for i, msg in enumerate(new_messages):
                self.message_mapping[start_idx + i] = msg
            
            # Update last indexed ID
            self.last_indexed_message_id = max([msg['id'] for msg in new_messages])
            
            # Save updated index
            if self.enable_persistence:
                all_messages = list(self.message_mapping.values())
                self._save_index_to_disk(all_messages)
            
            conversation_logger.log_system_event(
                "message_index_updated",
                f"Added {len(new_messages)} new messages to index"
            )
            
        except Exception as e:
            conversation_logger.log_error("message_index_update_failed", str(e), "Failed to update message index")
    
    def _save_index_to_disk(self, messages: List[Dict]):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.persistence_manager.index_file))
            
            # Save mapping
            with open(self.persistence_manager.mapping_file, 'w') as f:
                json.dump(self.message_mapping, f, indent=2, default=str)
            
            # Save metadata
            metadata = {
                "index_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "embedding_model": self.embedding_model_name,
                "message_count": len(messages),
                "last_indexed_message_id": self.last_indexed_message_id,
                "vector_dimension": self.index.d,
                "max_message_length": self.max_message_length,
                "index_build_time": self.index_build_time
            }
            
            with open(self.persistence_manager.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            conversation_logger.log_system_event(
                "message_index_saved",
                f"Saved message index with {len(messages)} messages"
            )
            
        except Exception as e:
            conversation_logger.log_error("message_index_save_failed", str(e), "Failed to save message index")
    
    def search_similar_messages(self, 
                               query: str, 
                               k: int = 5,
                               exclude_conversation_ids: List[int] = None,
                               min_similarity_score: float = 0.3,
                               max_age_days: Optional[int] = 30) -> List[Dict]:
        """
        Search for semantically similar messages
        
        Args:
            query: Search query
            k: Number of similar messages to return
            exclude_conversation_ids: Conversation IDs to exclude from results
            min_similarity_score: Minimum similarity score threshold
            max_age_days: Maximum age of messages in days (None = no limit)
            
        Returns:
            List of similar messages with similarity scores
        """
        try:
            if not self.index or self.index.ntotal == 0:
                return []
            
            start_time = time.time()
            
            # Update index with any new messages
            self._update_index_incrementally()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            
            # Search in FAISS (get more candidates for filtering)
            search_k = min(k * 3, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, search_k)
            
            # Process results
            similar_messages = []
            exclude_conversation_ids = exclude_conversation_ids or []
            cutoff_date = datetime.now() - timedelta(days=max_age_days) if max_age_days else None
            
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx == -1:
                    continue
                
                # Get message data
                message_data = self.message_mapping.get(idx)
                if not message_data:
                    continue
                
                # Calculate similarity score (convert L2 distance to similarity)
                similarity_score = max(0.0, 1.0 - (distance / 2.0))  # Normalize L2 distance
                
                # Apply filters
                if similarity_score < min_similarity_score:
                    continue
                
                if message_data['conversation_id'] in exclude_conversation_ids:
                    continue
                
                if cutoff_date and message_data.get('created_at'):
                    msg_date = datetime.fromisoformat(message_data['created_at'])
                    if msg_date < cutoff_date:
                        continue
                
                # Add to results
                result_message = message_data.copy()
                result_message['similarity_score'] = similarity_score
                result_message['search_rank'] = len(similar_messages) + 1
                result_message['l2_distance'] = float(distance)
                
                similar_messages.append(result_message)
                
                # Stop if we have enough results
                if len(similar_messages) >= k:
                    break
            
            search_time = time.time() - start_time
            
            # Log search results
            conversation_logger.log_system_event(
                "message_search_completed",
                f"Found {len(similar_messages)} similar messages for query '{query[:50]}...' in {search_time:.3f}s"
            )
            
            return similar_messages
            
        except Exception as e:
            conversation_logger.log_error("message_search_failed", str(e), f"Failed to search similar messages for query: {query[:100]}")
            return []
    
    def get_contextual_messages_for_response(self, user_query: str, current_conversation_id: int, max_context_pairs: int = 3) -> List[Dict]:
        """
        Get contextually relevant conversation pairs to enhance response generation
        
        Args:
            user_query: Current user query
            current_conversation_id: Current conversation ID to exclude
            max_context_pairs: Maximum conversation pairs to return
            
        Returns:
            List of conversation pairs (user + assistant) with context
        """
        try:
            # Search for similar user messages, excluding current conversation
            similar_user_messages = self.search_similar_messages(
                query=user_query,
                k=max_context_pairs * 2,  # Get more candidates to find pairs
                exclude_conversation_ids=[current_conversation_id],
                min_similarity_score=0.6,  # Much higher threshold to avoid poor matches
                max_age_days=7  # Only recent messages for relevance
            )
            
            # Filter to only user messages and find their assistant responses
            context_pairs = []
            
            for user_msg in similar_user_messages:
                if user_msg['role'] != 'user' or len(context_pairs) >= max_context_pairs:
                    continue
                
                # Find the corresponding assistant response
                assistant_response = self._find_assistant_response_for_user_message(user_msg['id'])
                
                if assistant_response:
                    conversation_pair = {
                        'user_message': user_msg['content'],
                        'assistant_response': assistant_response['content'],
                        'tool_name': assistant_response.get('tool_name'),
                        'tool_id': assistant_response.get('tool_id'),
                    }
                    context_pairs.append(conversation_pair)
            
            conversation_logger.log_system_event(
                "contextual_pairs_retrieved",
                f"Retrieved {len(context_pairs)} conversation pairs for response generation"
            )            
            return context_pairs
            
        except Exception as e:
            conversation_logger.log_error("contextual_pairs_failed", str(e), "Failed to get contextual conversation pairs")
            return []
    
    def _find_assistant_response_for_user_message(self, user_message_id: int) -> Optional[Dict]:
        """
        Find the assistant response that corresponds to a user message
        
        Args:
            user_message_id: ID of the user message
            
        Returns:
            Assistant response dictionary or None if not found
        """
        try:
            query = """
                SELECT id, content, created_at, metadata, tool_name, tool_id
                FROM messages
                WHERE parent_message_id = %s 
                AND role = 'assistant'
                ORDER BY sequence_number ASC
                LIMIT 1
            """
            
            result = db_manager.execute_query(query, (user_message_id,))
            
            if result:
                row = result[0]
                return {
                    'id': row[0],
                    'content': row[1],
                    'created_at': row[2].isoformat() if row[2] else None,
                    'metadata': row[3] if row[3] else {},  # JSONB is already a Python dict
                    'tool_name': row[4],  # NEW: Include tool name
                    'tool_id': row[5]     # NEW: Include tool ID
                }
            
            return None
            
        except Exception as e:
            conversation_logger.log_error("assistant_response_lookup_failed", str(e), f"Failed to find assistant response for message {user_message_id}")
            return None
    
    def add_message_to_index(self, message_id: int):
        """
        Add a single message to the index (called when new messages are created)
        
        Args:
            message_id: ID of the message to add
        """
        try:
            # This will be called by the next index update
            conversation_logger.log_system_event(
                "message_queued_for_indexing",
                f"Message {message_id} queued for next index update"
            )
            
        except Exception as e:
            conversation_logger.log_error("message_index_add_failed", str(e), f"Failed to queue message {message_id} for indexing")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about the message index"""
        try:
            stats = {
                "total_messages_indexed": len(self.message_mapping),
                "last_indexed_message_id": self.last_indexed_message_id,
                "index_dimension": self.index.d if self.index else 0,
                "embedding_model": self.embedding_model_name,
                "index_build_time": self.index_build_time,
                "persistence_enabled": self.enable_persistence
            }
            
            if self.message_mapping:
                # Message distribution by role
                roles = [msg['role'] for msg in self.message_mapping.values()]
                stats['message_roles'] = {role: roles.count(role) for role in set(roles)}
                
                # Conversation distribution
                conv_ids = [msg['conversation_id'] for msg in self.message_mapping.values()]
                stats['conversations_indexed'] = len(set(conv_ids))
            
            return stats
            
        except Exception as e:
            conversation_logger.log_error("message_index_stats_failed", str(e), "Failed to get message index statistics")
            return {"error": str(e)}
    
    def rebuild_index(self):
        """Force rebuild the entire message index"""
        try:
            conversation_logger.log_system_event("message_index_rebuild_start", "Starting manual message index rebuild")
            self.last_indexed_message_id = 0
            self.message_mapping = {}
            self._build_index_from_scratch()
            
        except Exception as e:
            conversation_logger.log_error("message_index_rebuild_failed", str(e), "Failed to rebuild message index")
            raise