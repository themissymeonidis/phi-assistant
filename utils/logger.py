import logging
import sys
from typing import Optional
from config import config

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Setup logger with custom formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        logger.setLevel(getattr(logging, level or config.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    return logger

class LocalAssistantError(Exception):
    """Base exception for Local Assistant"""
    pass

class ModelError(LocalAssistantError):
    """Model-related errors"""
    pass

class ToolError(LocalAssistantError):
    """Tool execution errors"""
    pass

class DatabaseError(LocalAssistantError):
    """Database-related errors"""
    pass

class ConfigurationError(LocalAssistantError):
    """Configuration errors"""
    pass
