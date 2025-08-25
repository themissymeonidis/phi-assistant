from model.model import Phi3Model
from embeddings.managers.tool_embedding import ToolEmbeddingManager
from embeddings.managers.message_embedding import MessageEmbeddingManager
from tools.tools import Tools
from utils.conversation_logger import conversation_logger
from utils.input_handler import InputHandler
from utils.conversation_history import ConversationHistoryManager
from commands.command_handler import CommandHandler
from services.tool_selection_service import ToolSelectionService
import time




class Orchestrator:
    def __init__(self):

        # Load Model
        self.model = Phi3Model();
        self.model.load_model();

        # Initialize tool embeddings manager
        self.tool_embeddings = ToolEmbeddingManager()
        
        # Initialize message embedding manager
        self.message_embeddings = MessageEmbeddingManager()

        # Load Tools
        self.tools = Tools()
        
        # Initialize input handler
        self.input_handler = InputHandler()
        
        # Initialize conversation history manager
        self.conversation_history = ConversationHistoryManager()

        # Initialize tool selection service
        self.tool_selection_service = ToolSelectionService(
            tool_embeddings=self.tool_embeddings,
            message_embeddings=self.message_embeddings,
            conversation_history=self.conversation_history
        )
        
        # Initialize command handler
        self.command_handler = CommandHandler(
            conversation_history=self.conversation_history,
            message_embeddings=self.message_embeddings,
            input_handler=self.input_handler,
            model=self.model
        )

        conversation_logger.log_system_event("session_started", "Orchestrator initialized and ready")
        print("🤖 Local Assistant Ready! Type 'help' for commands or ask me anything.")
        
        while True:
            # Get and validate user input
            input_data = self.input_handler.get_user_input("You: ")
            
            # Handle different input types
            if input_data['type'] == 'command':
                result = self.command_handler.handle_command(input_data['action'], input_data['text'])
                if result and result.get('should_exit'):
                    break
                continue
                
            elif input_data['type'] == 'invalid':
                self.command_handler.handle_invalid_input(input_data)
                continue
            
            # Process valid message
            user_input = input_data['text']
            conversation_logger.log_user_input(user_input)
            
            # Use tool selection service to find appropriate tool
            search_result = self.tool_selection_service.select_tool_with_context(user_input)
            
            if search_result['found_matching_tool']:
                # Execute tool directly since tool_id matches context
                tool_data = search_result['tool']
                
                conversation_logger.log_system_event(
                    "tool_match_found", 
                    f"Executing tool: {tool_data['name']} - tool_id matches context"
                )
                
                assistant_response = self._generate_and_store_response(
                    user_input,
                    None,
                    tool_data=tool_data
                )
                
            else:
                # No tool match found, generate regular response
                contextual_pairs = search_result.get('context', [])
                conversation_logger.log_system_event("no_tool_match", "No matching tools found, using regular chat")
                assistant_response = self._generate_and_store_response(
                    user_input,
                    None,
                    contextual_pairs=contextual_pairs
                )
    

    
    
    def _generate_and_store_response(self, user_input: str, response_generator, tool_data: dict = None, tool_name: str = None, tool_result: dict = None, tool_id: int = None, contextual_pairs = None) -> str:
        """
        Generate assistant response with contextual message history and store the complete conversation exchange
        
        Args:
            user_input: User's message
            response_generator: Function that generates the assistant response
            tool_name: Name of tool used (if any)
            tool_result: Result from tool execution (if any)
            tool_id: ID of tool used (if any)
            contextual_pairs: Pre-retrieved contextual pairs (if any)
            
        Returns:
            Generated assistant response text
        """
        try:
            # Execute tool if tool_data is provided
            if tool_data:
                function_name = tool_data["python_function"]
                method_to_call = getattr(self.tools, function_name)
                tool_result = method_to_call()
                tool_name = tool_data["name"]
                tool_id = tool_data["id"]
                
                # Log tool execution
                conversation_logger.log_tool_execution(
                    tool_data["name"], 
                    tool_result,
                )
                conversation_logger.log_system_event("tool_response_start", f"Using tool: {tool_data['name']}")

            # Store the user message first and get its ID
            user_message_id = self.conversation_history.add_message(
                role='user',
                content=user_input
            )
            
            # Determine what to pass to the model based on priority:
            # 1. If tool exists, pass tool context only
            # 2. If no tool but context exists, pass context only  
            # 3. Else pass nothing (direct generation)
            
            if tool_name and tool_result:
                # Tool execution - pass tool context only
                tool_context = {
                    'name': tool_name,
                    'result': tool_result
                }
                assistant_response = self.model.generate_with_context(
                    user_input=user_input,
                    tool_context=tool_context,
                    max_tokens=512,
                    temperature=0.7
                )
            elif contextual_pairs:
                # No tool but have context - pass context only
                assistant_response = self.model.generate_with_context(
                    user_input=user_input,
                    embedding_context=contextual_pairs,
                    max_tokens=512,
                    temperature=0.7
                )
            else:
                # No tool, no context - direct generation
                assistant_response = self.model.generate_with_context(
                    user_input=user_input,
                    max_tokens=512,
                    temperature=0.7
                )
            
            # Store tool response if a tool was used
            tool_message_id = None
            if tool_name and tool_result:
                tool_message_id = self.conversation_history.add_tool_response(
                    tool_name=tool_name,
                    tool_result=tool_result
                )
            
            # Store the assistant response with proper parent linking and context metadata
            metadata = {}          
            assistant_message_id = self.conversation_history.add_message(
                role='assistant',
                content=assistant_response,
                tool_name=tool_name if not tool_result else None,  # Only set if no separate tool message
                tool_id=tool_id,
                parent_message_id=user_message_id,  # Link to the user question
                metadata=metadata
            )
            
            conversation_logger.log_system_event(
                "conversation_stored", 
                f"Exchange stored: conversation_id={self.conversation_history.current_conversation_id}, "
                f"user_msg={user_message_id}, assistant_msg={assistant_message_id}, tool_msg={tool_message_id}, "
            )
            
            return assistant_response
            
        except Exception as e:
            conversation_logger.log_error("conversation_storage_failed", str(e), "Failed to store conversation exchange")
            # Still try to generate response even if storage fails
            try:
                return self.model.generate_with_context(user_input, max_tokens=512, temperature=0.7)
            except:
                return "Response generation failed"
    

orchestrator = Orchestrator();