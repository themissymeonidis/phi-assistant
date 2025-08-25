import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

class ConversationLogger:
    """Comprehensive logger for tracking the entire AI conversation flow"""
    
    def __init__(self, log_file="conversation.log"):
        self.logger = logging.getLogger("ConversationFlow")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for conversation logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler for important events (errors only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.session_start = time.time()
        self.logger.info("=== NEW CONVERSATION SESSION STARTED ===")
    
    def log_user_input(self, user_query: str):
        """Log user input"""
        self.logger.info(f"USER_INPUT | {user_query}")
    
    def log_faiss_search(self, query: str, results: List[Dict], search_time: float):
        """Log FAISS semantic search results"""
        result_summary = []
        try:
            for tool in results:
                result_summary.append({
                    "name": tool.get("name", "unknown"),
                    "distance": float(tool.get("distance", 0)),
                    "semantic_score": float(tool.get("semantic_score", 0))
                })
            
            self.logger.info(
                f"FAISS_SEARCH | Query: '{query}' | "
                f"Results: {json.dumps(result_summary)} | "
                f"Search_Time: {search_time:.3f}s"
            )
        except Exception as e:
            self.logger.error(f"FAISS_SEARCH_LOG_ERROR | Error logging search results: {e} | Query: '{query}'")
    
    def log_model_prompt(self, prompt_type: str, prompt: str, context: Optional[str] = None):
        """Log exact prompts sent to model"""
        log_entry = f"MODEL_PROMPT | Type: {prompt_type} | Prompt:\n{prompt}"
        if context:
            log_entry += f" | Context: {context}"
        self.logger.info(log_entry)
    
    def log_model_response(self, response_type: str, response: str, streaming: bool = False, tokens: int = None):
        """Log model responses"""
        log_entry = f"MODEL_RESPONSE | Type: {response_type} | Streaming: {streaming}"
        if tokens:
            log_entry += f" | Tokens: {tokens}"
        log_entry += f" | Response:\n{response}"
        self.logger.info(log_entry)
    
    def log_tool_evaluation(self, user_query: str, tool_name: str, decision: bool, confidence: float, reasoning: str, evaluation_time: float):
        """Log tool evaluation decisions with confidence"""
        self.logger.info(
            f"TOOL_EVALUATION | Query: '{user_query}' | "
            f"Tool: {tool_name} | Decision: {decision} | "
            f"Confidence: {confidence:.2f} | Reasoning: '{reasoning}' | "
            f"Eval_Time: {evaluation_time:.3f}s"
        )
    
    def log_tool_execution(self, tool_name: str, tool_result: Dict[str, Any], additional_metadata: Dict = None):
        """Log tool execution results with optional enhanced metadata"""
        log_message = f"TOOL_EXECUTION | Tool: {tool_name} | Result: {json.dumps(tool_result)}"
        
        if additional_metadata:
            metadata_str = " | ".join([f"{k}: {v}" for k, v in additional_metadata.items() if v is not None])
            if metadata_str:
                log_message += f" | {metadata_str}"
        
        self.logger.info(log_message)
    
    def log_context_management(self, action: str, details: str, total_tokens: int):
        """Log context window management events"""
        self.logger.info(
            f"CONTEXT_MGMT | Action: {action} | Details: {details} | "
            f"Total_Tokens: {total_tokens}"
        )
    
    def log_health_check(self, status: bool, details: str = ""):
        """Log model health check results"""
        status_str = "HEALTHY" if status else "UNHEALTHY"
        # Only log unhealthy status as warning, healthy status as info
        if status:
            self.logger.info(f"HEALTH_CHECK | Status: {status_str} | Details: {details}")
        else:
            self.logger.warning(f"HEALTH_CHECK | Status: {status_str} | Details: {details}")
    
    def log_retry_attempt(self, attempt: int, max_attempts: int, error: str):
        """Log retry attempts"""
        self.logger.warning(
            f"RETRY_ATTEMPT | Attempt: {attempt}/{max_attempts} | Error: {error}"
        )
    
    def log_conversation_metrics(self, total_exchanges: int, avg_response_time: float, total_tokens: int):
        """Log conversation performance metrics"""
        session_duration = time.time() - self.session_start
        self.logger.info(
            f"SESSION_METRICS | Duration: {session_duration:.1f}s | "
            f"Exchanges: {total_exchanges} | Avg_Response: {avg_response_time:.2f}s | "
            f"Total_Tokens: {total_tokens}"
        )
    
    def log_error(self, error_type: str, error_message: str, context: str = ""):
        """Log errors with context"""
        log_entry = f"ERROR | Type: {error_type} | Message: {error_message}"
        if context:
            log_entry += f" | Context: {context}"
        self.logger.error(log_entry)
    
    def log_system_event(self, event_type: str, details: str):
        """Log system events"""
        self.logger.info(f"SYSTEM_EVENT | Type: {event_type} | Details: {details}")

# Global conversation logger instance
conversation_logger = ConversationLogger()