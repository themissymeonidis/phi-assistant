import logging
import json
import time
from typing import Dict, Any, List, Optional
from .config import LoggerConfig

class Logger:
    """Comprehensive logger for tracking the entire AI conversation flow with separate log files"""
    
    def __init__(self, prompts_log_file=None, system_log_file=None, exceptions_log_file=None):
        # Use config defaults if not provided
        prompts_log_file = prompts_log_file or LoggerConfig.PROMPTS_LOG_FILE
        system_log_file = system_log_file or LoggerConfig.SYSTEM_LOG_FILE
        exceptions_log_file = exceptions_log_file or LoggerConfig.EXCEPTIONS_LOG_FILE
        # Initialize prompt logger for user/assistant interactions
        self.prompt_logger = logging.getLogger(LoggerConfig.PROMPTS_LOGGER_NAME)
        self.prompt_logger.setLevel(getattr(logging, LoggerConfig.DEFAULT_LOG_LEVEL))
        
        # Initialize system logger for other events
        self.system_logger = logging.getLogger(LoggerConfig.SYSTEM_LOGGER_NAME)
        self.system_logger.setLevel(getattr(logging, LoggerConfig.DEFAULT_LOG_LEVEL))
        
        # Initialize exceptions logger for errors and exceptions
        self.exceptions_logger = logging.getLogger(LoggerConfig.EXCEPTIONS_LOGGER_NAME)
        self.exceptions_logger.setLevel(getattr(logging, LoggerConfig.DEFAULT_LOG_LEVEL))
        
        # Create file handler for prompts
        prompts_handler = logging.FileHandler(prompts_log_file)
        prompts_handler.setLevel(getattr(logging, LoggerConfig.DEFAULT_LOG_LEVEL))
        
        # Create file handler for system events
        system_handler = logging.FileHandler(system_log_file)
        system_handler.setLevel(getattr(logging, LoggerConfig.DEFAULT_LOG_LEVEL))
        
        # Create file handler for exceptions
        exceptions_handler = logging.FileHandler(exceptions_log_file)
        exceptions_handler.setLevel(getattr(logging, LoggerConfig.DEFAULT_LOG_LEVEL))
        
        # Create console handler for important events (errors only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LoggerConfig.CONSOLE_LOG_LEVEL))
        
        # Create detailed formatter
        formatter = logging.Formatter(
            LoggerConfig.DEFAULT_FORMATTER,
            datefmt=LoggerConfig.DATE_FORMAT
        )
        
        prompts_handler.setFormatter(formatter)
        system_handler.setFormatter(formatter)
        exceptions_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.prompt_logger.addHandler(prompts_handler)
        self.system_logger.addHandler(system_handler)
        self.exceptions_logger.addHandler(exceptions_handler)
        self.exceptions_logger.addHandler(console_handler)
        
        self.session_start = time.time()
        self.system_logger.info(LoggerConfig.SESSION_START_MARKER)
    
    def log_user_input(self, user_query: str):
        """Log user input to prompts log"""
        self.prompt_logger.info(f"USER_INPUT | {user_query}")
    
    def log_model_prompt(self, prompt_type: str, prompt: str, context: Optional[str] = None):
        """Log exact prompts sent to model to prompts log"""
        log_entry = f"MODEL_PROMPT | Type: {prompt_type} | Prompt:\n{prompt}"
        if context:
            log_entry += f" | Context: {context}"
        self.prompt_logger.info(log_entry)
    
    def log_model_response(self, response_type: str, response: str, streaming: bool = False, tokens: int = None):
        """Log model responses to prompts log"""
        log_entry = f"MODEL_RESPONSE | Type: {response_type} | Streaming: {streaming}"
        if tokens:
            log_entry += f" | Tokens: {tokens}"
        log_entry += f" | Response:\n{response}"
        self.prompt_logger.info(log_entry)
    
    def log_faiss_search(self, query: str, results: List[Dict], search_time: float):
        """Log FAISS semantic search results to system log"""
        result_summary = []
        try:
            for tool in results:
                result_summary.append({
                    "name": tool.get("name", "unknown"),
                    "distance": float(tool.get("distance", 0)),
                    "semantic_score": float(tool.get("semantic_score", 0))
                })
            
            self.system_logger.info(
                f"FAISS_SEARCH | Query: '{query}' | "
                f"Results: {json.dumps(result_summary)} | "
                f"Search_Time: {search_time:.3f}s"
            )
        except Exception as e:
            self.log_exception("FAISS_SEARCH_LOG_ERROR", str(e), f"Query: '{query}'")
    
    def log_tool_evaluation(self, user_query: str, tool_name: str, decision: bool, confidence: float, reasoning: str, evaluation_time: float):
        """Log tool evaluation decisions with confidence to system log"""
        self.system_logger.info(
            f"TOOL_EVALUATION | Query: '{user_query}' | "
            f"Tool: {tool_name} | Decision: {decision} | "
            f"Confidence: {confidence:.2f} | Reasoning: '{reasoning}' | "
            f"Eval_Time: {evaluation_time:.3f}s"
        )
    
    def log_tool_execution(self, tool_name: str, tool_result: Dict[str, Any], additional_metadata: Dict = None):
        """Log tool execution results with optional enhanced metadata to system log"""
        log_message = f"TOOL_EXECUTION | Tool: {tool_name} | Result: {json.dumps(tool_result)}"
        
        if additional_metadata:
            metadata_str = " | ".join([f"{k}: {v}" for k, v in additional_metadata.items() if v is not None])
            if metadata_str:
                log_message += f" | {metadata_str}"
        
        self.system_logger.info(log_message)
    
    def log_context_management(self, action: str, details: str, total_tokens: int):
        """Log context window management events to system log"""
        self.system_logger.info(
            f"CONTEXT_MGMT | Action: {action} | Details: {details} | "
            f"Total_Tokens: {total_tokens}"
        )
    
    def log_health_check(self, status: bool, details: str = ""):
        """Log model health check results to system log"""
        status_str = "HEALTHY" if status else "UNHEALTHY"
        if status:
            self.system_logger.info(f"HEALTH_CHECK | Status: {status_str} | Details: {details}")
        else:
            self.system_logger.warning(f"HEALTH_CHECK | Status: {status_str} | Details: {details}")
    
    def log_retry_attempt(self, attempt: int, max_attempts: int, error: str):
        """Log retry attempts to system log"""
        self.system_logger.warning(
            f"RETRY_ATTEMPT | Attempt: {attempt}/{max_attempts} | Error: {error}"
        )
    
    def log_conversation_metrics(self, total_exchanges: int, avg_response_time: float, total_tokens: int):
        """Log conversation performance metrics to system log"""
        session_duration = time.time() - self.session_start
        self.system_logger.info(
            f"SESSION_METRICS | Duration: {session_duration:.1f}s | "
            f"Exchanges: {total_exchanges} | Avg_Response: {avg_response_time:.2f}s | "
            f"Total_Tokens: {total_tokens}"
        )
    
    def log_error(self, error_type: str, error_message: str, context: str = ""):
        """Log errors with context to exceptions log"""
        log_entry = f"ERROR | Type: {error_type} | Message: {error_message}"
        if context:
            log_entry += f" | Context: {context}"
        self.exceptions_logger.error(log_entry)
    
    def log_exception(self, exception_type: str, exception_message: str, context: str = ""):
        """Log exceptions with context to exceptions log"""
        log_entry = f"EXCEPTION | Type: {exception_type} | Message: {exception_message}"
        if context:
            log_entry += f" | Context: {context}"
        self.exceptions_logger.error(log_entry)
    
    def log_system_event(self, event_type: str, details: str):
        """Log system events to system log"""
        self.system_logger.info(f"SYSTEM_EVENT | Type: {event_type} | Details: {details}")

# Global logger instance
logger = Logger()