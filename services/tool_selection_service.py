"""
Tool Selection Service
Handles the logic for selecting and matching tools based on user input and context
"""

from typing import Dict, List, Optional
from utils.conversation_logger import conversation_logger


class ToolSelectionService:
    """
    Service for intelligent tool selection based on user input and conversation context
    """
    
    def __init__(self, tool_embeddings, message_embeddings, conversation_history):
        """
        Initialize the tool selection service
        
        Args:
            tool_embeddings: ToolEmbeddingManager instance
            message_embeddings: MessageEmbeddingManager instance
            conversation_history: ConversationHistoryManager instance
        """
        self.tool_embeddings = tool_embeddings
        self.message_embeddings = message_embeddings
        self.conversation_history = conversation_history
    
    def select_tool_with_context(self, user_input: str) -> Dict:
        """
        Select appropriate tool based on user input and conversation context
        
        Args:
            user_input: User's query
            
        Returns:
            Dictionary with selection results:
            {
                'found_matching_tool': bool,
                'tool': dict (if found),
                'context': list (contextual pairs),
                'selection_reason': str
            }
        """
        try:
            # Get available tools
            tool_candidates = self.tool_embeddings.query_tools_optimized(
                user_input, 
                max_candidates=10,
                min_semantic_score=0.3
            )
            
            # Get context from similar conversations
            contextual_pairs = self.message_embeddings.get_contextual_messages_for_response(
                user_query=user_input,
                current_conversation_id=self.conversation_history.current_conversation_id or 0,
                max_context_pairs=3
            )
            
            # Check if any tool matches context tool_id
            for tool in tool_candidates:
                tool_id = tool.get('id')
                for context in contextual_pairs:
                    if context.get('tool_id') == tool_id:
                        conversation_logger.log_system_event(
                            "tool_selection_match_found",
                            f"Tool '{tool['name']}' matched context tool_id {tool_id}"
                        )
                        return {
                            'found_matching_tool': True,
                            'tool': tool,
                            'context': contextual_pairs,
                            'selection_reason': f"Tool '{tool['name']}' matched historical context"
                        }
            
            # No matching tool found
            conversation_logger.log_system_event(
                "tool_selection_no_match",
                f"No tool matched context for query: {user_input[:50]}..."
            )
            return {
                'found_matching_tool': False,
                'context': contextual_pairs,
                'selection_reason': "No tool matched conversation context"
            }
            
        except Exception as e:
            conversation_logger.log_error(
                "tool_selection_failed", 
                str(e), 
                f"Tool selection failed for query: {user_input[:100]}"
            )
            return {
                'found_matching_tool': False,
                'context': [],
                'selection_reason': f"Error during tool selection: {str(e)}"
            }
    
    def get_tool_candidates(self, user_input: str, max_candidates: int = 10) -> List[Dict]:
        """
        Get tool candidates for user input
        
        Args:
            user_input: User's query
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of tool candidates with scores
        """
        return self.tool_embeddings.query_tools_optimized(
            user_input,
            max_candidates=max_candidates,
            min_semantic_score=0.3
        )
    
    def get_contextual_pairs(self, user_input: str, max_pairs: int = 3) -> List[Dict]:
        """
        Get contextual conversation pairs for user input
        
        Args:
            user_input: User's query
            max_pairs: Maximum number of context pairs to return
            
        Returns:
            List of contextual conversation pairs
        """
        return self.message_embeddings.get_contextual_messages_for_response(
            user_query=user_input,
            current_conversation_id=self.conversation_history.current_conversation_id or 0,
            max_context_pairs=max_pairs
        )
