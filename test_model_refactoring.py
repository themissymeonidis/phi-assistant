#!/usr/bin/env python3
"""
Test script to verify the model refactoring works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_imports():
    """Test that the refactored model can be imported"""
    try:
        from model.model import Phi3Model
        print("✅ Model imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Model import error: {e}")
        return False

def test_model_initialization():
    """Test that the model can be initialized with new parameters"""
    try:
        from model.model import Phi3Model
        
        # Test with default parameters
        model = Phi3Model()
        print("✅ Model initialization with defaults successful!")
        
        # Test with custom model path
        model = Phi3Model(model_path="/custom/path/model.gguf")
        print("✅ Model initialization with custom path successful!")
        
        return True
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False

def test_context_aware_generation():
    """Test the new context-aware generation method"""
    try:
        from model.model import Phi3Model
        
        model = Phi3Model()
        
        # Mock conversation context
        conversation_context = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'How are you?'}
        ]
        
        # Mock embedding context
        embedding_context = [
            {
                'user_message': {'original_content': 'What is Python?'},
                'assistant_response': {'content': 'Python is a programming language.'}
            }
        ]
        
        # Mock tool context
        tool_context = {
            'name': 'calculator',
            'result': {'result': 42}
        }
        
        # Test the method exists and can be called (without actually loading the model)
        if hasattr(model, 'generate_with_context'):
            print("✅ generate_with_context method exists!")
        else:
            print("❌ generate_with_context method not found")
            return False
        
        if hasattr(model, '_build_contextual_prompt'):
            print("✅ _build_contextual_prompt method exists!")
        else:
            print("❌ _build_contextual_prompt method not found")
            return False
        
        # Test prompt building
        prompt = model._build_contextual_prompt(
            user_input="Test query",
            conv_context=conversation_context,
            embed_context=embedding_context,
            tool_context=tool_context
        )
        
        if "Test query" in prompt and "Recent Conversation Context" in prompt:
            print("✅ Contextual prompt building works!")
        else:
            print("❌ Contextual prompt building failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Context-aware generation test error: {e}")
        return False

def test_removed_methods():
    """Test that redundant methods have been removed"""
    try:
        from model.model import Phi3Model
        
        model = Phi3Model()
        
        # Check that redundant methods are removed
        if hasattr(model, 'manage_context_window'):
            print("❌ manage_context_window method still exists (should be removed)")
            return False
        
        if hasattr(model, 'chat'):
            print("❌ chat method still exists (should be replaced)")
            return False
        
        if hasattr(model, 'conversation_history'):
            print("❌ conversation_history attribute still exists (should be removed)")
            return False
        
        print("✅ Redundant methods successfully removed!")
        return True
    except Exception as e:
        print(f"❌ Removed methods test error: {e}")
        return False

def test_error_handling():
    """Test that common error handling method exists"""
    try:
        from model.model import Phi3Model
        
        model = Phi3Model()
        
        if hasattr(model, '_handle_generation_error'):
            print("✅ Common error handling method exists!")
        else:
            print("❌ Common error handling method not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Model Refactoring...")
    print("=" * 50)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("Model Initialization", test_model_initialization),
        ("Context-Aware Generation", test_context_aware_generation),
        ("Removed Methods", test_removed_methods),
        ("Error Handling", test_error_handling),
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
        print("🎉 All tests passed! Model refactoring successful!")
        return 0
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
