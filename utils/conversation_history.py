"""
Conversation History Manager for Local Assistant
Handles persistent storage of conversations and messages in PostgreSQL
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from utils.database import db_manager
from utils.conversation_logger import conversation_logger


class ConversationHistoryManager:
    """
    Manages persistent conversation history in PostgreSQL database
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.current_conversation_id: Optional[int] = None
        self.current_sequence_number = 0
        
    def start_new_conversation(self, title: str = "Untitled Conversation", metadata: Optional[Dict] = None) -> int:
        """
        Start a new conversation and return its ID
        
        Args:
            title: Title for the conversation
            metadata: Additional metadata to store
            
        Returns:
            conversation_id: ID of the created conversation
        """
        try:
            insert_sql = """
                INSERT INTO conversations (title, session_id, metadata) 
                VALUES (%s, %s, %s) 
                RETURNING id
            """
            
            metadata_json = json.dumps(metadata or {})
            result = db_manager.execute_query(insert_sql, (title, self.session_id, metadata_json))
            
            if result and len(result) > 0:
                conversation_id = result[0][0]
                self.current_conversation_id = conversation_id
                self.current_sequence_number = 0
                
                conversation_logger.log_system_event(
                    "conversation_started", 
                    f"New conversation created: {conversation_id} - {title}"
                )
                
                return conversation_id
            else:
                raise Exception("Failed to get conversation ID from database")
                
        except Exception as e:
            conversation_logger.log_error("conversation_start_failed", str(e), "Failed to start new conversation")
            raise
    
    def add_message(self, 
                   role: str, 
                   content: str, 
                   tool_name: Optional[str] = None,
                   tool_result: Optional[Dict] = None,
                   tool_id: Optional[int] = None,
                   is_correction: bool = False,
                   parent_message_id: Optional[int] = None,
                   metadata: Optional[Dict] = None) -> int:
        """
        Add a message to the current conversation
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            tool_name: Name of tool used (if applicable)
            tool_result: Structured tool result data
            tool_id: ID of tool used (if applicable)
            is_correction: Whether this message is a correction
            parent_message_id: ID of parent message (for threading)
            metadata: Additional metadata
            
        Returns:
            message_id: ID of the created message
        """
        if not self.current_conversation_id:
            # Auto-start conversation if none exists
            title = self._generate_conversation_title(content)
            self.start_new_conversation(title)
        
        try:
            # Get next sequence number
            self.current_sequence_number += 1
            
            insert_sql = """
                INSERT INTO messages 
                (conversation_id, role, content, tool_name, tool_result, tool_id, is_correction, 
                 parent_message_id, sequence_number, metadata) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                RETURNING id
            """
            
            # Convert tool_result and metadata to JSON
            tool_result_json = json.dumps(tool_result) if tool_result else None
            metadata_json = json.dumps(metadata or {})
            
            result = db_manager.execute_query(insert_sql, (
                self.current_conversation_id,
                role,
                content,
                tool_name,
                tool_result_json,
                tool_id,
                is_correction,
                parent_message_id,
                self.current_sequence_number,
                metadata_json
            ))
            
            if result and len(result) > 0:
                message_id = result[0][0]
                
                conversation_logger.log_system_event(
                    "message_stored", 
                    f"Message {message_id} stored: {role} - {content[:50]}..."
                )
                
                return message_id
            else:
                raise Exception("Failed to get message ID from database")
                
        except Exception as e:
            conversation_logger.log_error("message_store_failed", str(e), "Failed to store message")
            raise
    
    def get_conversation_history(self, conversation_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve conversation history
        
        Args:
            conversation_id: Specific conversation ID (uses current if None)
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        target_conversation_id = conversation_id or self.current_conversation_id
        
        if not target_conversation_id:
            return []
        
        try:
            query_sql = """
                SELECT 
                    m.id, m.role, m.content, m.tool_name, m.tool_result, m.tool_id,
                    m.is_correction, m.sequence_number, m.created_at, m.metadata
                FROM messages m
                WHERE m.conversation_id = %s
                ORDER BY m.sequence_number ASC
                LIMIT %s
            """
            
            result = db_manager.execute_query(query_sql, (target_conversation_id, limit))
            
            messages = []
            for row in result:
                message = {
                    'id': row[0],
                    'role': row[1],
                    'content': row[2],
                    'tool_name': row[3],
                    'tool_result': row[4] if row[4] else None,  # JSONB is already a Python dict
                    'tool_id': row[5],
                    'is_correction': row[6],
                    'sequence_number': row[7],
                    'created_at': row[8].isoformat() if row[8] else None,
                    'metadata': row[9] if row[9] else {}  # JSONB is already a Python dict
                }
                messages.append(message)
            
            return messages
            
        except Exception as e:
            conversation_logger.log_error("conversation_retrieval_failed", str(e), "Failed to retrieve conversation history")
            return []
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversations summary
        
        Args:
            limit: Number of conversations to retrieve
            
        Returns:
            List of conversation summaries
        """
        try:
            query_sql = """
                SELECT 
                    id, title, summary, tool_usage_summary, started_at, ended_at, session_id,
                    message_count, last_message_at, metadata
                FROM conversation_summary
                ORDER BY started_at DESC
                LIMIT %s
            """
            
            result = db_manager.execute_query(query_sql, (limit,))
            
            conversations = []
            for row in result:
                conversation = {
                    'id': row[0],
                    'title': row[1],
                    'summary': row[2],
                    'tool_usage_summary': row[3],
                    'started_at': row[4].isoformat() if row[4] else None,
                    'ended_at': row[5].isoformat() if row[5] else None,
                    'session_id': row[6],
                    'message_count': row[7],
                    'last_message_at': row[8].isoformat() if row[8] else None,
                    'metadata': row[9] if row[9] else {}  # JSONB is already a Python dict
                }
                conversations.append(conversation)
            
            return conversations
            
        except Exception as e:
            conversation_logger.log_error("conversation_list_failed", str(e), "Failed to retrieve conversation list")
            return []
    
    def end_current_conversation(self, summary: Optional[str] = None) -> bool:
        """
        End the current conversation
        
        Args:
            summary: Optional summary to add to metadata
            
        Returns:
            Success status
        """
        if not self.current_conversation_id:
            return True
        
        try:
            metadata = {}
            if summary:
                metadata['summary'] = summary
            
            update_sql = """
                UPDATE conversations 
                SET ended_at = CURRENT_TIMESTAMP, metadata = metadata || %s
                WHERE id = %s
            """
            
            db_manager.execute_query(update_sql, (json.dumps(metadata), self.current_conversation_id))
            
            conversation_logger.log_system_event(
                "conversation_ended", 
                f"Conversation {self.current_conversation_id} ended"
            )
            
            self.current_conversation_id = None
            self.current_sequence_number = 0
            
            return True
            
        except Exception as e:
            conversation_logger.log_error("conversation_end_failed", str(e), "Failed to end conversation")
            return False
    
    def add_correction(self, original_message_id: int, corrected_content: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a correction message
        
        Args:
            original_message_id: ID of the message being corrected
            corrected_content: Corrected content
            metadata: Additional metadata
            
        Returns:
            ID of the correction message
        """
        correction_metadata = metadata or {}
        correction_metadata['original_message_id'] = original_message_id
        
        return self.add_message(
            role='user',
            content=corrected_content,
            is_correction=True,
            parent_message_id=original_message_id,
            metadata=correction_metadata
        )
    
    def add_tool_response(self, tool_name: str, tool_result: Dict, content: str = None) -> int:
        """
        Add a tool response message
        
        Args:
            tool_name: Name of the executed tool
            tool_result: Structured result from the tool
            content: Optional human-readable content
            
        Returns:
            ID of the tool message
        """
        if not content:
            content = f"Tool '{tool_name}' executed successfully"
        
        return self.add_message(
            role='tool',
            content=content,
            tool_name=tool_name,
            tool_result=tool_result
        )
    
    def _generate_conversation_title(self, first_message: str, max_length: int = 50) -> str:
        """
        Generate a conversation title from the first message
        
        Args:
            first_message: First user message
            max_length: Maximum title length
            
        Returns:
            Generated title
        """
        # Simple title generation - could be enhanced with ML
        title = first_message.strip()
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        # Remove special characters and newlines
        title = ' '.join(title.split())
        
        return title or "Untitled Conversation"
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about conversation patterns
        
        Returns:
            Analytics dictionary
        """
        try:
            analytics_sql = """
                SELECT 
                    COUNT(DISTINCT c.id) as total_conversations,
                    COUNT(m.id) as total_messages,
                    AVG(message_counts.msg_count) as avg_messages_per_conversation,
                    COUNT(CASE WHEN m.role = 'tool' THEN 1 END) as tool_messages,
                    COUNT(CASE WHEN m.is_correction THEN 1 END) as correction_messages
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                LEFT JOIN (
                    SELECT conversation_id, COUNT(*) as msg_count 
                    FROM messages 
                    GROUP BY conversation_id
                ) message_counts ON c.id = message_counts.conversation_id
                WHERE c.session_id = %s
            """
            
            result = db_manager.execute_query(analytics_sql, (self.session_id,))
            
            if result and len(result) > 0:
                row = result[0]
                return {
                    'total_conversations': row[0] or 0,
                    'total_messages': row[1] or 0,
                    'avg_messages_per_conversation': float(row[2] or 0),
                    'tool_messages': row[3] or 0,
                    'correction_messages': row[4] or 0,
                    'session_id': self.session_id
                }
            
            return {}
            
        except Exception as e:
            conversation_logger.log_error("analytics_failed", str(e), "Failed to generate conversation analytics")
            return {}
    
    def process_conversation_exchange(self, user_input: str, assistant_response: str, 
                                    tool_name: Optional[str] = None, tool_result: Optional[Dict] = None,
                                    is_correction: bool = False) -> Dict[str, int]:
        """
        Process a complete conversation exchange (user input + assistant response)
        
        Args:
            user_input: User's message
            assistant_response: Assistant's response
            tool_name: Tool used (if any)
            tool_result: Tool result (if any)
            is_correction: Whether user input is a correction
            
        Returns:
            Dictionary with conversation_id and message_ids
        """
        try:
            # Add user message
            user_message_id = self.add_message(
                role='user',
                content=user_input,
                is_correction=is_correction
            )
            
            message_ids = [user_message_id]
            
            # Add tool message if tool was used
            if tool_name and tool_result:
                tool_message_id = self.add_tool_response(tool_name, tool_result)
                message_ids.append(tool_message_id)
            
            # Add assistant response
            assistant_message_id = self.add_message(
                role='assistant',
                content=assistant_response,
                tool_name=tool_name if not tool_result else None  # Only set tool_name if no separate tool message
            )
            message_ids.append(assistant_message_id)
            
            return {
                'conversation_id': self.current_conversation_id,
                'message_ids': message_ids,
                'user_message_id': user_message_id,
                'assistant_message_id': assistant_message_id,
                'tool_message_id': message_ids[1] if len(message_ids) > 2 else None
            }
            
        except Exception as e:
            conversation_logger.log_error("exchange_processing_failed", str(e), "Failed to process conversation exchange")
            raise
    
    def get_conversation_for_summary(self, conversation_id: int) -> Dict:
        """
        Get complete conversation data for summarization
        
        Args:
            conversation_id: ID of conversation to retrieve
            
        Returns:
            Dictionary with messages and tool usage
        """
        try:
            # Get conversation messages
            query = """
                SELECT id, role, content, tool_name, tool_result, created_at
                FROM messages
                WHERE conversation_id = %s
                AND role IN ('user', 'assistant', 'tool')
                ORDER BY sequence_number ASC
            """
            
            result = db_manager.execute_query(query, (conversation_id,))
            
            messages = []
            tools_used = []
            
            for row in result:
                msg_data = {
                    'id': row[0],
                    'role': row[1], 
                    'content': row[2],
                    'tool_name': row[3],
                    'tool_result': row[4],
                    'created_at': row[5].isoformat() if row[5] else None
                }
                
                if row[1] == 'tool':
                    # Track tool usage
                    tools_used.append({
                        'name': row[3],
                        'result': row[4] if row[4] else {}
                    })
                else:
                    # Add to messages (user/assistant only)
                    messages.append(msg_data)
            
            return {
                'conversation_id': conversation_id,
                'messages': messages,
                'tools_used': tools_used
            }
            
        except Exception as e:
            conversation_logger.log_error("conversation_summary_fetch_failed", str(e), f"Failed to fetch conversation {conversation_id} for summary")
            return {'conversation_id': conversation_id, 'messages': [], 'tools_used': []}
    
    def save_conversation_summary(self, conversation_id: int, title: str, summary: str, tool_usage_summary: str) -> bool:
        """
        Save generated conversation summary to database
        
        Args:
            conversation_id: ID of conversation
            title: Generated conversation title 
            summary: Generated conversation summary
            tool_usage_summary: Generated tool usage summary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
                UPDATE conversations 
                SET title = %s, summary = %s, tool_usage_summary = %s
                WHERE id = %s
            """
            
            affected_rows = db_manager.execute_command(
                query, 
                (title, summary, tool_usage_summary, conversation_id)
            )
            
            if affected_rows > 0:
                conversation_logger.log_system_event(
                    "conversation_summary_saved",
                    f"Saved summary for conversation {conversation_id}: '{title}'"
                )
                return True
            else:
                conversation_logger.log_error(
                    "conversation_summary_save_failed",
                    f"No conversation found with ID {conversation_id}",
                    "Update affected 0 rows"
                )
                return False
                
        except Exception as e:
            conversation_logger.log_error("conversation_summary_save_failed", str(e), f"Failed to save summary for conversation {conversation_id}")
            return False