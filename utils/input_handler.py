import re
from utils.conversation_logger import conversation_logger

class InputHandler:
    """Handle and validate user input with proper sanitization"""
    
    def __init__(self):
        self.min_input_length = 2
        self.max_input_length = 1000  # Prevent extremely long inputs
        self.exit_commands = ['exit', 'quit', 'bye', 'goodbye']
        self.help_commands = ['help', '/help', '?']
        self.clear_commands = ['clear', '/clear', 'reset']
        # Add new command patterns
        self.command_patterns = {
            'history': ['history', '/history'],
            'conversations': ['conversations', '/conversations'],
            'stats': ['stats', '/stats'],
            'search': ['search', '/search'],
            'embeddings': ['embeddings', '/embeddings'],
            'rebuild': ['rebuild', '/rebuild'],
            'summarise_conv': ['summarise_conv', '/summarise_conv', 'summarize_conv', '/summarize_conv', 'summarise', '/summarise']
        }
        
    def get_user_input(self, prompt="You: ") -> dict:
        """Get and validate user input - Returns dict with:
        - 'text': cleaned input text
        - 'type': 'message', 'command', 'exit', 'invalid'
        - 'action': specific action for commands
        """
        try:
            raw_input = input(prompt)
            return self.process_input(raw_input)
        except (KeyboardInterrupt, EOFError):
            conversation_logger.log_system_event("keyboard_interrupt", "User interrupted with Ctrl+C")
            return {'text': '', 'type': 'exit', 'action': 'interrupt'}
    
    def process_input(self, raw_input: str) -> dict:
        """Process and validate input"""
        
        # Basic sanitization
        cleaned_input = self.sanitize_input(raw_input)
        
        # Check for empty input
        if not cleaned_input:
            conversation_logger.log_system_event("empty_input", "User provided empty input")
            return {'text': '', 'type': 'invalid', 'action': 'empty'}
        
        # Check input length
        if len(cleaned_input) > self.max_input_length:
            conversation_logger.log_system_event("input_too_long", f"Input length: {len(cleaned_input)}")
            return {'text': cleaned_input[:self.max_input_length], 'type': 'invalid', 'action': 'too_long'}
        
        # Check for exit commands
        if cleaned_input.lower() in self.exit_commands:
            return {'text': cleaned_input, 'type': 'command', 'action': 'exit'}
        
        # Check for help commands
        if cleaned_input.lower() in self.help_commands:
            return {'text': cleaned_input, 'type': 'command', 'action': 'help'}
        
        # Check for clear commands
        if cleaned_input.lower() in self.clear_commands:
            return {'text': cleaned_input, 'type': 'command', 'action': 'clear'}
        
        # Check for other commands (exact match and with parameters)
        for action, patterns in self.command_patterns.items():
            if cleaned_input.lower() in patterns:
                return {'text': cleaned_input, 'type': 'command', 'action': action}
            
            # Check for commands with parameters (e.g., "summarise_conv 25")
            for pattern in patterns:
                if cleaned_input.lower().startswith(pattern + ' '):
                    return {'text': cleaned_input, 'type': 'command', 'action': action}
        
        # Check minimum length for regular messages
        if len(cleaned_input) < self.min_input_length:
            conversation_logger.log_system_event("input_too_short", f"Input: '{cleaned_input}'")
            return {'text': cleaned_input, 'type': 'invalid', 'action': 'too_short'}
        
        # Valid message input
        return {'text': cleaned_input, 'type': 'message', 'action': 'process'}
    
    def sanitize_input(self, raw_input: str) -> str:
        """Clean and sanitize user input"""
        if not isinstance(raw_input, str):
            return ""
        
        # Strip whitespace
        cleaned = raw_input.strip()
        
        # Remove null bytes and control characters (except newlines/tabs)
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
        
        # Normalize whitespace (replace multiple spaces with single space)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def get_help_text(self) -> str:
        """Return help text for user"""
        return """
Available commands:
• Type your question or message normally
• 'exit', 'quit' - Exit the application
• 'clear' - Clear conversation history  
• 'help', '?' - Show this help message

Tips:
• Messages must be at least 2 characters long
• Maximum message length: 1000 characters
• Use Ctrl+C to interrupt at any time
        """.strip()
    
    def handle_invalid_input(self, input_data: dict) -> str:
        """Provide user feedback for invalid input"""
        action = input_data.get('action', 'unknown')
        
        if action == 'empty':
            return "⚠️  Please enter a message or question."
        elif action == 'too_short':
            return f"⚠️  Please enter at least {self.min_input_length} characters."
        elif action == 'too_long':
            return f"⚠️  Message too long! Maximum {self.max_input_length} characters allowed."
        else:
            return "⚠️  Invalid input. Type 'help' for assistance."