import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Dict, Any
from config import config
from utils.logger import setup_logger, DatabaseError

logger = setup_logger(__name__)

class DatabaseManager:
    """Manages database connections with pooling"""
    
    def __init__(self):
        self.pool: Optional[pool.SimpleConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.user,
                password=config.database.password
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseError(f"Database connection failed: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:
        """Execute a query and return results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            try:
                return cursor.fetchall()
            except psycopg2.ProgrammingError:
                # No results to fetch (e.g., UPDATE, INSERT without RETURNING)
                return []
    
    def execute_command(self, command: str, params: Optional[tuple] = None) -> int:
        """Execute a command and return affected rows"""
        with self.get_cursor() as cursor:
            cursor.execute(command, params)
            return cursor.rowcount
    
    def close(self):
        """Close the connection pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

# Global database manager instance
db_manager = DatabaseManager()
