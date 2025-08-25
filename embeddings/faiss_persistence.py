import faiss
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from utils.conversation_logger import conversation_logger
from utils.database import db_manager


class FaissPersistenceManager:
    """
    Handles FAISS index persistence, validation, and synchronization with database
    """
    
    def __init__(self, index_dir: str = "./embeddings/indexes"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / "tools.faiss"
        self.metadata_file = self.index_dir / "tools_metadata.json"
        self.mapping_file = self.index_dir / "tools_mapping.json"
        
    def calculate_tools_checksum(self, tools_data: Dict) -> str:
        """
        Calculate SHA256 checksum of tools data for change detection
        
        Args:
            tools_data: Dictionary of tool data from database
            
        Returns:
            SHA256 hex digest of tools data
        """
        # Create deterministic representation of tools data
        tools_list = []
        for name in sorted(tools_data.keys()):
            tool = tools_data[name]
            # Include fields that affect embeddings/search
            tool_repr = f"{tool['id']}|{name}|{tool['description']}|{tool['query_examples']}"
            tools_list.append(tool_repr)
        
        combined_data = "|".join(tools_list)
        return hashlib.sha256(combined_data.encode('utf-8')).hexdigest()
    
    def get_database_last_update(self) -> Optional[str]:
        """
        Get the most recent updated_at timestamp from tools table
        
        Returns:
            ISO format timestamp string or None if no tools found
        """
        try:
            query = """
            SELECT MAX(updated_at) FROM tools WHERE active = TRUE
            """
            result = db_manager.execute_query(query)
            
            if result and result[0][0]:
                # Convert to ISO format string
                return result[0][0].isoformat()
            return None
            
        except Exception as e:
            conversation_logger.log_error("db_timestamp_query_failed", str(e), "Failed to query database timestamps")
            return None
    
    def save_index_with_metadata(self, 
                                index: faiss.Index, 
                                tool_mapping: Dict, 
                                tools_data: Dict,
                                embedding_model: str = "all-MiniLM-L6-v2") -> bool:
        """
        Save FAISS index and associated metadata to disk
        
        Args:
            index: FAISS index to save
            tool_mapping: Mapping from index positions to tool data
            tools_data: Original tools data from database
            embedding_model: Name of embedding model used
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Save FAISS index
            faiss.write_index(index, str(self.index_file))
            
            # Save tool mapping
            with open(self.mapping_file, 'w') as f:
                json.dump(tool_mapping, f, indent=2)
            
            # Create and save metadata
            metadata = {
                "index_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "tools_count": len(tools_data),
                "tools_checksum": self.calculate_tools_checksum(tools_data),
                "embedding_model": embedding_model,
                "index_type": type(index).__name__,
                "vector_dimension": index.d,
                "last_db_update": self.get_database_last_update(),
                "index_file_size": os.path.getsize(self.index_file),
                "mapping_file_size": os.path.getsize(self.mapping_file)
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            conversation_logger.log_system_event(
                "faiss_index_saved", 
                f"Saved index with {len(tools_data)} tools to {self.index_file}"
            )
            
            return True
            
        except Exception as e:
            conversation_logger.log_error("faiss_index_save_failed", str(e), "Failed to save FAISS index")
            return False
    
    def load_index_with_validation(self, tools_data: Dict, embedding_model: str = "all-MiniLM-L6-v2") -> Tuple[Optional[faiss.Index], Optional[Dict], bool]:
        """
        Load FAISS index from disk with comprehensive validation
        
        Args:
            tools_data: Current tools data from database for validation
            embedding_model: Expected embedding model name
            
        Returns:
            Tuple of (index, tool_mapping, is_valid)
            - index: Loaded FAISS index or None if invalid
            - tool_mapping: Tool mapping dict or None if invalid  
            - is_valid: True if index is current and valid
        """
        try:
            # Check if all required files exist
            if not all(f.exists() for f in [self.index_file, self.metadata_file, self.mapping_file]):
                conversation_logger.log_system_event("faiss_index_missing", "Index files not found, will rebuild")
                return None, None, False
            
            # Load and validate metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate metadata integrity
            validation_issues = []
            
            # Check embedding model consistency
            if metadata.get("embedding_model") != embedding_model:
                validation_issues.append(f"Embedding model mismatch: expected {embedding_model}, got {metadata.get('embedding_model')}")
            
            # Check tools count
            if metadata.get("tools_count") != len(tools_data):
                validation_issues.append(f"Tools count mismatch: expected {len(tools_data)}, got {metadata.get('tools_count')}")
            
            # Check tools data checksum
            current_checksum = self.calculate_tools_checksum(tools_data)
            if metadata.get("tools_checksum") != current_checksum:
                validation_issues.append(f"Tools data changed: checksum mismatch")
            
            # Check file sizes for corruption detection
            if os.path.getsize(self.index_file) != metadata.get("index_file_size", 0):
                validation_issues.append("Index file size mismatch - possible corruption")
            
            if os.path.getsize(self.mapping_file) != metadata.get("mapping_file_size", 0):
                validation_issues.append("Mapping file size mismatch - possible corruption")
            
            # Check database update timestamp
            current_db_update = self.get_database_last_update()
            if current_db_update and metadata.get("last_db_update") != current_db_update:
                validation_issues.append(f"Database updated since index creation")
            
            # If validation issues found, return invalid
            if validation_issues:
                conversation_logger.log_system_event(
                    "faiss_index_invalid", 
                    f"Index validation failed: {'; '.join(validation_issues)}"
                )
                return None, None, False
            
            # Load index and mapping
            index = faiss.read_index(str(self.index_file))
            
            with open(self.mapping_file, 'r') as f:
                tool_mapping = json.load(f)
            
            # Final validation: check index dimensions
            expected_dimension = metadata.get("vector_dimension", 384)
            if index.d != expected_dimension:
                validation_issues.append(f"Index dimension mismatch: expected {expected_dimension}, got {index.d}")
                return None, None, False
            
            # Success - index is valid and current
            conversation_logger.log_system_event(
                "faiss_index_loaded", 
                f"Successfully loaded valid index with {len(tool_mapping)} tools"
            )
            
            return index, tool_mapping, True
            
        except Exception as e:
            conversation_logger.log_error("faiss_index_load_failed", str(e), "Failed to load FAISS index")
            return None, None, False
    
    def cleanup_old_indexes(self, keep_backups: int = 2) -> None:
        """
        Clean up old index files, keeping only recent backups
        
        Args:
            keep_backups: Number of backup versions to keep
        """
        try:
            # Look for backup files (if implemented)
            backup_pattern = "tools_backup_*.faiss"
            backup_files = list(self.index_dir.glob(backup_pattern))
            
            if len(backup_files) > keep_backups:
                # Sort by modification time and remove oldest
                backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for old_backup in backup_files[keep_backups:]:
                    old_backup.unlink()
                    conversation_logger.log_system_event("faiss_backup_cleaned", f"Removed old backup: {old_backup.name}")
                    
        except Exception as e:
            conversation_logger.log_error("faiss_cleanup_failed", str(e), "Failed to cleanup old index files")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the persisted index
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "index_exists": self.index_file.exists(),
            "metadata_exists": self.metadata_file.exists(), 
            "mapping_exists": self.mapping_file.exists()
        }
        
        if stats["metadata_exists"]:
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                stats.update({
                    "created_at": metadata.get("created_at"),
                    "tools_count": metadata.get("tools_count"),
                    "embedding_model": metadata.get("embedding_model"),
                    "index_version": metadata.get("index_version"),
                    "index_file_size_mb": round(metadata.get("index_file_size", 0) / 1024 / 1024, 2)
                })
            except Exception:
                stats["metadata_corrupt"] = True
        
        return stats