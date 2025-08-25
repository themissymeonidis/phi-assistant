"""
Conversation History Demo - Demonstrates the conversation persistence system
"""

from utils.conversation_history import ConversationHistoryManager
from utils.conversation_logger import conversation_logger
import json

def demonstrate_conversation_scenario():
    """
    Demonstrate the example scenario from the requirements:
    
    User: "What is the weather in Athens?"
    Assistant: "It seems sunny, but I didn't check."
    User: "Please use the weather tool for that question."
    """
    
    print("🎯 Conversation History Demo")
    print("=" * 50)
    
    # Initialize conversation manager
    conversation_mgr = ConversationHistoryManager()
    
    print("Step 1: Starting new conversation...")
    conversation_id = conversation_mgr.start_new_conversation(
        title="Weather Query with Correction",
        metadata={"demo": True, "scenario": "weather_correction"}
    )
    print(f"✅ Created conversation ID: {conversation_id}")
    
    print("\nStep 2: Adding user message...")
    user_msg_1_id = conversation_mgr.add_message(
        role='user',
        content="What is the weather in Athens?"
    )
    print(f"✅ User message ID: {user_msg_1_id}")
    
    print("\nStep 3: Adding assistant response...")
    assistant_msg_1_id = conversation_mgr.add_message(
        role='assistant',
        content="It seems sunny, but I didn't check.",
        metadata={"confidence": 0.3, "tool_used": False}
    )
    print(f"✅ Assistant message ID: {assistant_msg_1_id}")
    
    print("\nStep 4: Adding correction message...")
    correction_msg_id = conversation_mgr.add_correction(
        original_message_id=assistant_msg_1_id,
        corrected_content="Please use the weather tool for that question.",
        metadata={"correction_type": "tool_request"}
    )
    print(f"✅ Correction message ID: {correction_msg_id}")
    
    print("\nStep 5: Adding tool response...")
    tool_msg_id = conversation_mgr.add_tool_response(
        tool_name="get_weather",
        tool_result={
            "location": "Athens",
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "45%",
            "timestamp": "2024-01-15T14:30:00Z"
        },
        content="Weather tool executed successfully for Athens"
    )
    print(f"✅ Tool message ID: {tool_msg_id}")
    
    print("\nStep 6: Adding corrected assistant response...")
    assistant_msg_2_id = conversation_mgr.add_message(
        role='assistant',
        content="Based on the weather tool, Athens is currently 22°C and sunny with 45% humidity.",
        tool_name="get_weather",
        metadata={"confidence": 0.95, "tool_used": True}
    )
    print(f"✅ Final assistant message ID: {assistant_msg_2_id}")
    
    print("\n📊 Final Results:")
    print("-" * 30)
    
    # Show conversation history
    history = conversation_mgr.get_conversation_history()
    print(f"Total messages in conversation: {len(history)}")
    
    for i, msg in enumerate(history, 1):
        role_icon = {'user': '👤', 'assistant': '🤖', 'tool': '🔧'}.get(msg['role'], '❓')
        correction_flag = " ⚠️ CORRECTION" if msg['is_correction'] else ""
        print(f"{i}. {role_icon} {msg['role'].upper()}{correction_flag}: {msg['content'][:60]}...")
        if msg['tool_name']:
            print(f"   🔧 Tool: {msg['tool_name']}")
    
    # Show SQL statements that were executed
    print("\n🗄️ Database Operations Summary:")
    print("-" * 40)
    print("1. INSERT INTO conversations (title, session_id, metadata) VALUES (...)")
    print("2. INSERT INTO messages (role='user', content='What is the weather...')")  
    print("3. INSERT INTO messages (role='assistant', content='It seems sunny...')")
    print("4. INSERT INTO messages (role='user', is_correction=TRUE, content='Please use...')")
    print("5. INSERT INTO messages (role='tool', tool_name='get_weather', content='Weather tool...')")
    print("6. INSERT INTO messages (role='assistant', tool_name='get_weather', content='Based on...')")
    
    # End conversation
    print("\nStep 7: Ending conversation...")
    conversation_mgr.end_current_conversation("Demo completed successfully")
    print("✅ Conversation ended")
    
    # Show analytics
    print("\n📈 Session Analytics:")
    stats = conversation_mgr.get_conversation_analytics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return {
        'conversation_id': conversation_id,
        'message_ids': [user_msg_1_id, assistant_msg_1_id, correction_msg_id, tool_msg_id, assistant_msg_2_id],
        'sql_operations': 6,
        'conversation_ended': True
    }

def show_sql_examples():
    """Show the exact SQL statements that would be generated"""
    
    print("\n🔍 SQL Statement Examples:")
    print("=" * 50)
    
    # Conversation creation
    print("1. Creating new conversation:")
    print("""
    INSERT INTO conversations (title, session_id, metadata) 
    VALUES ('Weather Query with Correction', 'session_abc123', '{"demo": true}') 
    RETURNING id;
    """)
    
    # User message
    print("2. Adding user message:")
    print("""
    INSERT INTO messages 
    (conversation_id, role, content, sequence_number, metadata) 
    VALUES (1, 'user', 'What is the weather in Athens?', 1, '{}') 
    RETURNING id;
    """)
    
    # Assistant response
    print("3. Adding assistant response:")
    print("""
    INSERT INTO messages 
    (conversation_id, role, content, sequence_number, metadata) 
    VALUES (1, 'assistant', 'It seems sunny, but I didn't check.', 2, '{"confidence": 0.3}') 
    RETURNING id;
    """)
    
    # Correction
    print("4. Adding correction:")
    print("""
    INSERT INTO messages 
    (conversation_id, role, content, is_correction, parent_message_id, sequence_number, metadata) 
    VALUES (1, 'user', 'Please use the weather tool for that question.', TRUE, 2, 3, '{"original_message_id": 2}') 
    RETURNING id;
    """)
    
    # Tool response
    print("5. Adding tool response:")
    print("""
    INSERT INTO messages 
    (conversation_id, role, content, tool_name, tool_result, sequence_number) 
    VALUES (1, 'tool', 'Weather tool executed successfully', 'get_weather', 
            '{"location": "Athens", "temperature": "22°C", "condition": "Sunny"}', 4) 
    RETURNING id;
    """)
    
    # Final response
    print("6. Adding final assistant response:")
    print("""
    INSERT INTO messages 
    (conversation_id, role, content, tool_name, sequence_number, metadata) 
    VALUES (1, 'assistant', 'Based on the weather tool, Athens is currently 22°C...', 
            'get_weather', 5, '{"confidence": 0.95, "tool_used": true}') 
    RETURNING id;
    """)

def json_response_example():
    """Show the JSON response format"""
    
    print("\n📄 JSON Response Example:")
    print("=" * 30)
    
    response = {
        "conversation_id": 1,
        "message_ids": [1, 2, 3, 4, 5],
        "user_message_id": 1,
        "assistant_message_id": 2,
        "correction_message_id": 3,
        "tool_message_id": 4,
        "final_response_id": 5,
        "sql_operations": [
            "INSERT conversation -> ID: 1",
            "INSERT user message -> ID: 1", 
            "INSERT assistant message -> ID: 2",
            "INSERT correction -> ID: 3",
            "INSERT tool response -> ID: 4",
            "INSERT final response -> ID: 5"
        ],
        "status": "success"
    }
    
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    try:
        print("🚀 Starting Conversation History Demonstration")
        
        # Run the demo scenario
        result = demonstrate_conversation_scenario()
        
        # Show SQL examples
        show_sql_examples()
        
        # Show JSON response
        json_response_example()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Conversation ID: {result['conversation_id']}")
        print(f"   Messages created: {len(result['message_ids'])}")
        print(f"   SQL operations: {result['sql_operations']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        conversation_logger.log_error("demo_failed", str(e), "Conversation demo encountered an error")