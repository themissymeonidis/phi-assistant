#!/usr/bin/env python3
"""
Test script to verify the command extraction refactoring works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all the new modules can be imported correctly"""
    try:
        from commands.command_handler import CommandHandler
        from utils.input_handler import InputHandler
        from utils.conversation_history import ConversationHistoryManager
        from embeddings.message_embeddings import MessageEmbeddingManager
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_command_handler_creation():
    """Test that CommandHandler can be instantiated"""
    try:
        from commands.command_handler import CommandHandler
        from utils.input_handler import InputHandler
        from utils.conversation_history import ConversationHistoryManager
        from embeddings.message_embeddings import MessageEmbeddingManager
        
        # Mock objects for testing
        class MockConversationHistory:
            def get_conversation_history(self, limit):
                return []
            def get_recent_conversations(self, limit):
                return []
            def get_conversation_analytics(self):
                return {}
            current_conversation_id = 0
        
        class MockMessageEmbeddings:
            def search_similar_messages(self, query, k, exclude_conversation_ids, min_similarity_score):
                return []
            def get_index_statistics(self):
                return {'total_messages_indexed': 0}
            def rebuild_index(self):
                pass
        
        input_handler = InputHandler()
        conversation_history = MockConversationHistory()
        message_embeddings = MockMessageEmbeddings()
        
        command_handler = CommandHandler(
            conversation_history=conversation_history,
            message_embeddings=message_embeddings,
            input_handler=input_handler
        )
        
        print("✅ CommandHandler instantiation successful!")
        return True
    except Exception as e:
        print(f"❌ CommandHandler creation error: {e}")
        return False

def test_input_handler_commands():
    """Test that InputHandler recognizes the new commands"""
    try:
        from utils.input_handler import InputHandler
        
        input_handler = InputHandler()
        
        # Test command recognition
        test_commands = ['history', '/history', 'stats', '/stats', 'search', '/search', 'exit', 'quit', 'bye']
        
        for cmd in test_commands:
            result = input_handler.process_input(cmd)
            if result['type'] != 'command':
                print(f"❌ Command '{cmd}' not recognized as command")
                return False
        
        print("✅ InputHandler command recognition successful!")
        return True
    except Exception as e:
        print(f"❌ InputHandler command test error: {e}")
        return False

def test_exit_command():
    """Test that exit command returns proper exit status"""
    try:
        from commands.command_handler import CommandHandler
        from utils.input_handler import InputHandler
        
        # Mock objects for testing
        class MockConversationHistory:
            def end_current_conversation(self, reason):
                pass
        
        class MockMessageEmbeddings:
            def search_similar_messages(self, query, k, exclude_conversation_ids, min_similarity_score):
                return []
            def get_index_statistics(self):
                return {'total_messages_indexed': 0}
            def rebuild_index(self):
                pass
        
        input_handler = InputHandler()
        conversation_history = MockConversationHistory()
        message_embeddings = MockMessageEmbeddings()
        
        command_handler = CommandHandler(
            conversation_history=conversation_history,
            message_embeddings=message_embeddings,
            input_handler=input_handler
        )
        
        # Test exit command
        result = command_handler.handle_command('exit')
        if result and result.get('should_exit'):
            print("✅ Exit command returns proper exit status!")
            return True
        else:
            print("❌ Exit command does not return proper exit status")
            return False
            
    except Exception as e:
        print(f"❌ Exit command test error: {e}")
        return False

def test_invalid_input_handling():
    """Test that invalid input handling works through CommandHandler"""
    try:
        from commands.command_handler import CommandHandler
        from utils.input_handler import InputHandler
        
        # Mock objects for testing
        class MockConversationHistory:
            def end_current_conversation(self, reason):
                pass
        
        class MockMessageEmbeddings:
            def search_similar_messages(self, query, k, exclude_conversation_ids, min_similarity_score):
                return []
            def get_index_statistics(self):
                return {'total_messages_indexed': 0}
            def rebuild_index(self):
                pass
        
        input_handler = InputHandler()
        conversation_history = MockConversationHistory()
        message_embeddings = MockMessageEmbeddings()
        
        command_handler = CommandHandler(
            conversation_history=conversation_history,
            message_embeddings=message_embeddings,
            input_handler=input_handler
        )
        
        # Test invalid input handling
        invalid_input = {'type': 'invalid', 'action': 'too_short', 'text': 'a'}
        command_handler.handle_invalid_input(invalid_input)
        
        print("✅ Invalid input handling works through CommandHandler!")
        return True
            
    except Exception as e:
        print(f"❌ Invalid input handling test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Command System Extraction...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("CommandHandler Creation", test_command_handler_creation),
        ("InputHandler Commands", test_input_handler_commands),
        ("Exit Command", test_exit_command),
        ("Invalid Input Handling", test_invalid_input_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Command extraction refactoring successful!")
        return 0
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
