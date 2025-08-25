import numpy as np
from sentence_transformers import SentenceTransformer
import faiss, time
from terminal.animations import Animations
from utils.conversation_logger import conversation_logger
from utils.database import db_manager
from embeddings.faiss_persistence import FaissPersistenceManager

from dotenv import load_dotenv
load_dotenv()



class ToolEmbeddingsManager:
    def __init__(self, distance_threshold=1.5, enable_persistence=True):
        self.animator = Animations()
        self.distance_threshold = distance_threshold
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.enable_persistence = enable_persistence
        
        # Initialize persistence manager
        if self.enable_persistence:
            self.persistence_manager = FaissPersistenceManager()
        
        def load():
            self._load_with_persistence()
        self.animator.run_with_animation(load, message="Loading Vector Indexes...")


    def prepare_embeddings(self):
        # Handle query_examples whether it's a string or list
        descriptions = []
        for tool in self.tool_dict.values():
            query_examples = tool["query_examples"]
            if isinstance(query_examples, list):
                # Join list items with space for embedding
                descriptions.append(" ".join(query_examples))
            else:
                descriptions.append(query_examples)
        
        vectors = self.embedding_model.encode(descriptions)  # shape (n, d)
        vectors = np.array(vectors, dtype='float32')
        mapping = {i: {"name": name, **self.tool_dict[name]} for i, name in enumerate(self.tool_dict)}
        return vectors, mapping

    def create_faiss_index(self, vectors):
        d = vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        return index
    
    def query_tools_optimized(self, user_query, max_candidates=3, min_semantic_score=0.4):
        """
        Optimized tool querying with precision-focused filtering
        
        Args:
            user_query: User's input query
            max_candidates: Maximum tools to return (hard limit)
            min_semantic_score: Minimum score threshold before LLM evaluation
        
        Returns:
            List of top-ranked tools with enhanced filtering
        """
        start_time = time.time()
        
        # Phase 1: Conservative FAISS search
        query_vector = self.embedding_model.encode([user_query]).astype('float32')
        
        # Search more candidates initially for better ranking
        search_k = min(15, len(self.tool_mapping) * 2)
        distances, indices = self.index.search(query_vector, search_k)
        
        # Phase 2: Multi-factor scoring and filtering
        candidates = []
        conservative_threshold = 0.8  # Much stricter than original 1.5
        
        for i, dist in zip(indices[0], distances[0]):
            if i == -1 or dist > conservative_threshold:
                continue
                
            tool_data = self.tool_mapping[i].copy()
            
            # Semantic similarity (primary factor)
            semantic_score = max(0.0, 1.0 - (dist / conservative_threshold))
            
            # Skip low-scoring candidates early
            if semantic_score < min_semantic_score:
                continue
                
            # Query-tool length matching
            query_tokens = len(user_query.split())
            # Handle query_examples whether it's a string or list
            query_examples = tool_data['query_examples']
            if isinstance(query_examples, list):
                tool_tokens = sum(len(example.split()) for example in query_examples)
            else:
                tool_tokens = len(query_examples.split())
            description_tokens = len(tool_data['description'].split())
            
            # Length similarity factor (optimal range: 0.3-3.0 ratio)
            length_ratio = query_tokens / max(1, tool_tokens)
            length_score = 1.0 - abs(1.0 - min(3.0, max(0.33, length_ratio))) / 2.0
            
            # Description relevance (longer descriptions may be more specific)
            description_factor = min(1.0, description_tokens / max(1, query_tokens))
            
            # Keyword overlap bonus
            query_words = set(user_query.lower().split())
            # Handle query_examples whether it's a string or list for keyword matching
            query_examples = tool_data['query_examples']
            if isinstance(query_examples, list):
                tool_words = set()
                for example in query_examples:
                    tool_words.update(example.lower().split())
            else:
                tool_words = set(query_examples.lower().split())
            overlap_ratio = len(query_words & tool_words) / max(1, len(query_words))
            keyword_bonus = overlap_ratio * 0.2
            
            # Combined scoring with weights
            combined_score = (
                0.50 * semantic_score +           # Primary: semantic similarity
                0.25 * length_score +             # Secondary: length matching  
                0.15 * description_factor +       # Tertiary: description depth
                0.10 * keyword_bonus              # Bonus: direct keyword matches
            )
            
            tool_data.update({
                'semantic_score': semantic_score,
                'length_score': length_score,
                'description_factor': description_factor,
                'keyword_bonus': keyword_bonus,
                'combined_score': combined_score,
                'distance': float(dist)
            })
            
            candidates.append(tool_data)
        
        # Phase 3: Ranking and selection
        if not candidates:
            # Fallback: try with relaxed threshold
            return self._fallback_search(user_query, min_semantic_score * 0.5, max_candidates)
        
        # Sort by combined score and apply top_k limit
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        final_candidates = candidates[:max_candidates]
        
        # Phase 4: Quality gates
        filtered_candidates = []
        for candidate in final_candidates:
            # High-confidence bypass (skip some LLM evaluation steps)
            if candidate['combined_score'] >= 0.8:
                candidate['llm_priority'] = 'high_confidence'
            elif candidate['combined_score'] >= 0.6:
                candidate['llm_priority'] = 'standard'
            else:
                candidate['llm_priority'] = 'low_confidence'
                
            # Additional quality gate: ensure semantic score meets minimum
            if candidate['semantic_score'] >= min_semantic_score:
                filtered_candidates.append(candidate)
        
        search_time = time.time() - start_time
        
        # Enhanced logging with filtering statistics
        conversation_logger.log_faiss_search(
            user_query, 
            filtered_candidates, 
            search_time
        )
        
        return filtered_candidates
    
    def query_tools_with_ranking(self, user_query, k=5, semantic_weight=0.6):
        """Enhanced tool querying with multiple ranking factors"""
        query_vector = self.embedding_model.encode([user_query]).astype('float32')
        distances, indices = self.index.search(query_vector, k * 2)  # Get more candidates

        candidates = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1 and dist <= self.distance_threshold * 1.2:  # Slightly more permissive
                tool_data = self.tool_mapping[i].copy()
                
                # Semantic similarity score (0-1, higher is better)
                semantic_score = max(0.0, 1.0 - (dist / (self.distance_threshold * 1.2)))
                
                # Query length matching factor (longer descriptions might be more detailed)
                query_len = len(user_query.split())
                # Handle query_examples whether it's a string or list
                query_examples = tool_data['query_examples']
                if isinstance(query_examples, list):
                    desc_len = sum(len(example.split()) for example in query_examples)
                else:
                    desc_len = len(query_examples.split())
                length_factor = min(1.0, query_len / max(1, desc_len)) if desc_len > 0 else 0.5
                
                # Combined score
                combined_score = (semantic_weight * semantic_score + 
                                (1 - semantic_weight) * length_factor)
                
                tool_data.update({
                    'semantic_score': semantic_score,
                    'length_factor': length_factor,
                    'combined_score': combined_score,
                    'distance': float(dist)
                })
                
                candidates.append(tool_data)

        # Sort by combined score and return top k
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:k]

    def load_db_tools(self):
        """Load tools from database using centralized database manager"""
        query = "SELECT id, name, description, python_function, query_examples FROM tools WHERE active = TRUE"
        
        try:
            results = db_manager.execute_query(query)
            conversation_logger.log_system_event("db_tools_loaded", f"Loaded {len(results)} active tools from database")
            
            return {
                row[1]: {  # row[1] is the 'name' column
                    "id": row[0], 
                    "description": row[2], 
                    "python_function": row[3], 
                    "query_examples": row[4]
                } 
                for row in results
            }
            
        except Exception as e:
            conversation_logger.log_error("db_tools_load_failed", str(e), "Failed to load tools from database")
            # Return empty dict as fallback
            return {}
    
    def _fallback_search(self, user_query, relaxed_threshold, max_candidates):
        """Fallback search with very relaxed thresholds when no good matches found"""
        query_vector = self.embedding_model.encode([user_query]).astype('float32')
        distances, indices = self.index.search(query_vector, max_candidates * 2)
        
        fallback_candidates = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1 and dist <= 1.2:  # Still stricter than original 1.5
                tool_data = self.tool_mapping[i].copy()
                semantic_score = max(0.0, 1.0 - (dist / 1.2))
                
                if semantic_score >= relaxed_threshold:
                    tool_data.update({
                        'semantic_score': semantic_score,
                        'combined_score': semantic_score,
                        'distance': float(dist),
                        'llm_priority': 'fallback'
                    })
                    fallback_candidates.append(tool_data)
        
        return fallback_candidates[:max_candidates]
    
    def should_skip_llm_evaluation(self, tool_candidate, user_query):
        """
        Determine if LLM evaluation can be skipped for high-confidence matches
        
        Returns:
            bool: True if tool should be executed without LLM evaluation
        """
        # Very high confidence + keyword match = skip LLM
        if (tool_candidate.get('combined_score', 0) >= 0.85 and 
            tool_candidate.get('keyword_bonus', 0) >= 0.1):
            return True
            
        # Perfect semantic match for simple queries
        if (len(user_query.split()) <= 5 and 
            tool_candidate.get('semantic_score', 0) >= 0.9):
            return True
            
        return False
    
    def _load_with_persistence(self):
        """
        Load index with persistence support and fallback mechanisms
        """
        # Load tools from database
        self.tool_dict = self.load_db_tools()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        if not self.enable_persistence or not self.tool_dict:
            # Fallback to direct rebuild
            conversation_logger.log_system_event("tool_embeddings_persistence_disabled", "Building index from scratch")
            self._build_index_from_scratch()
            return
        
        # Try to load persisted index
        loaded_index, loaded_mapping, is_valid = self.persistence_manager.load_index_with_validation(
            self.tool_dict, 
            self.embedding_model_name
        )
        
        if is_valid and loaded_index and loaded_mapping:
            # Use persisted index
            self.index = loaded_index
            self.tool_mapping = loaded_mapping
            
            # Convert string keys back to integers for tool_mapping
            self.tool_mapping = {int(k): v for k, v in loaded_mapping.items()}
            
            conversation_logger.log_system_event(
                "tool_embeddings_loaded_from_disk", 
                f"Loaded persisted index with {len(self.tool_mapping)} tools"
            )
            
        else:
            # Rebuild and save
            conversation_logger.log_system_event(
                "tool_embeddings_rebuild_required", 
                "Rebuilding index due to validation failure or missing files"
            )
            self._build_index_from_scratch()
            self._save_index_to_disk()
    
    def _build_index_from_scratch(self):
        """
        Build FAISS index from scratch by generating embeddings
        """
        self.tool_vectors, self.tool_mapping = self.prepare_embeddings()
        self.index = self.create_faiss_index(self.tool_vectors)
        
    def _save_index_to_disk(self):
        """
        Save current index to disk with metadata
        """
        if self.enable_persistence and hasattr(self, 'index') and hasattr(self, 'tool_mapping'):
            success = self.persistence_manager.save_index_with_metadata(
                self.index,
                self.tool_mapping,
                self.tool_dict,
                self.embedding_model_name
            )
            
            if success:
                conversation_logger.log_system_event(
                    "tool_embeddings_persisted", 
                    f"Successfully persisted index with {len(self.tool_dict)} tools"
                )
            else:
                conversation_logger.log_error(
                    "tool_embeddings_persistence_failed", 
                    "Failed to persist index to disk", 
                    "Index will be rebuilt on next startup"
                )
    
    def rebuild_index(self, save_to_disk=True):
        """
        Force rebuild the tool embeddings index (useful for manual updates)
        
        Args:
            save_to_disk: Whether to persist the rebuilt index
        """
        conversation_logger.log_system_event("tool_embeddings_manual_rebuild", "Manually rebuilding tool embeddings index")
        
        # Reload tools from database
        self.tool_dict = self.load_db_tools()
        
        # Rebuild index
        self._build_index_from_scratch()
        
        # Save to disk if requested
        if save_to_disk and self.enable_persistence:
            self._save_index_to_disk()
        
        conversation_logger.log_system_event(
            "tool_embeddings_rebuild_complete", 
            f"Index rebuilt with {len(self.tool_dict)} tools"
        )
    
    def get_index_statistics(self):
        """
        Get comprehensive statistics about the current index
        
        Returns:
            Dictionary with index and persistence statistics
        """
        stats = {
            "tools_count": len(self.tool_dict) if hasattr(self, 'tool_dict') else 0,
            "index_size": self.index.ntotal if hasattr(self, 'index') else 0,
            "vector_dimension": self.index.d if hasattr(self, 'index') else 0,
            "embedding_model": self.embedding_model_name,
            "persistence_enabled": self.enable_persistence
        }
        
        if self.enable_persistence:
            persistence_stats = self.persistence_manager.get_index_stats()
            stats.update({"persistence": persistence_stats})
        
        return stats
    
    def validate_index_sync(self):
        """
        Validate if the current index is synchronized with the database
        
        Returns:
            Tuple of (is_synchronized, issues_found)
        """
        if not self.enable_persistence:
            return True, ["Persistence disabled"]
        
        try:
            # Get current database state
            current_tools = self.load_db_tools()
            current_checksum = self.persistence_manager.calculate_tools_checksum(current_tools)
            
            # Compare with persisted metadata if available
            if self.persistence_manager.metadata_file.exists():
                import json
                with open(self.persistence_manager.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                stored_checksum = metadata.get('tools_checksum')
                if stored_checksum != current_checksum:
                    return False, ["Database tools have been modified since index creation"]
                
                if metadata.get('tools_count') != len(current_tools):
                    return False, ["Number of tools has changed"]
                
                return True, []
            else:
                return False, ["No metadata file found"]
                
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def get_performance_metrics(self):
        """
        Get performance metrics for monitoring and optimization
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = {
            "startup_mode": "persisted" if (hasattr(self, 'index') and 
                          self.enable_persistence and 
                          self.persistence_manager.index_file.exists()) else "rebuilt",
            "index_load_time": getattr(self, '_index_load_time', 0),
            "tools_count": len(self.tool_dict) if hasattr(self, 'tool_dict') else 0
        }
        
        return metrics
    
    def query_tools(self, user_query, k=3):
        """Legacy method - calls optimized version for backward compatibility"""
        return self.query_tools_optimized(user_query, max_candidates=k, min_semantic_score=0.4)