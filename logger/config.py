"""
Logger Configuration
Contains all static variables and settings for the logging system
"""

class LoggerConfig:
    """Configuration class for logging system"""
    
    # Log file names
    PROMPTS_LOG_FILE = "prompts.log"
    SYSTEM_LOG_FILE = "system.log" 
    EXCEPTIONS_LOG_FILE = "exceptions.log"
    
    # Log levels
    DEFAULT_LOG_LEVEL = "INFO"
    CONSOLE_LOG_LEVEL = "ERROR"
    
    # Log formatters
    DEFAULT_FORMATTER = "%(asctime)s | %(levelname)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Logger names
    PROMPTS_LOGGER_NAME = "ConversationPrompts"
    SYSTEM_LOGGER_NAME = "ConversationSystem"
    EXCEPTIONS_LOGGER_NAME = "ConversationExceptions"
    
    # Session markers
    SESSION_START_MARKER = "=== NEW CONVERSATION SESSION STARTED ==="
    SESSION_END_MARKER = "=== CONVERSATION SESSION ENDED ==="