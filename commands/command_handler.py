import time
from logger import logger
from utils.input_handler import InputHandler


class CommandHandler:
    """Handle user commands and system operations"""
    
    def __init__(self, conversation_history, message_embeddings, input_handler, model=None):
        self.conversation_history = conversation_history
        self.message_embeddings = message_embeddings
        self.input_handler = input_handler
        self.model = model
    
    def handle_command(self, action: str, full_input: str = None):
        """Route commands to appropriate handlers"""
        if action == 'help':
            self._show_help()
        elif action == 'clear':
            self._clear_conversation_history()
        elif action == 'history':
            self._show_conversation_history()
        elif action == 'conversations':
            self._show_recent_conversations()
        elif action == 'stats':
            self._show_conversation_stats()
        elif action == 'search':
            self._search_similar_messages()
        elif action == 'embeddings':
            self._show_embedding_stats()
        elif action == 'rebuild':
            self._rebuild_message_index()
        elif action == 'summarise_conv':
            self._summarise_conversation(full_input)
        elif action == 'exit':
            return self._handle_exit()
        else:
            print(f"❌ Unknown command: {action}")
            return None
    
    def handle_invalid_input(self, input_data: dict):
        """Handle invalid input and provide user feedback"""
        feedback = self.input_handler.handle_invalid_input(input_data)
        print(feedback)
        return None
    
    def _show_help(self):
        """Show help text with all available commands"""
        help_text = self.input_handler.get_help_text()
        # Add conversation commands to help
        help_text += "\n\nConversation Commands:"
        help_text += "\n  exit, quit, bye - Exit the application"
        help_text += "\n  history - Show recent conversation history"
        help_text += "\n  conversations - List recent conversations"
        help_text += "\n  stats - Show conversation statistics"
        help_text += "\n  search - Search for similar messages"
        help_text += "\n  embeddings - Show message embedding statistics"
        help_text += "\n  rebuild - Rebuild message embedding index"
        help_text += "\n  summarise_conv [ID] - Summarise a conversation by ID (interactive if no ID)"
        print(help_text)
        logger.log_system_event("help_requested", "User requested help")
    
    def _clear_conversation_history(self):
        """Clear the current conversation history"""
        if self.model and hasattr(self.model, 'conversation_history'):
            self.model.conversation_history.clear()
        print("🧹 Conversation history cleared!")
        logger.log_system_event("history_cleared", "User cleared conversation history")
    
    def _show_conversation_history(self):
        """Show current conversation history"""
        try:
            history = self.conversation_history.get_conversation_history(limit=20)
            
            if not history:
                print("📝 No conversation history found.")
                return
            
            print(f"📝 Current Conversation History ({len(history)} messages):")
            print("-" * 50)
            
            for msg in history:
                timestamp = msg['created_at'][:19] if msg['created_at'] else 'Unknown'
                role_emoji = {
                    'user': '👤',
                    'assistant': '🤖', 
                    'tool': '🔧',
                    'system': '⚙️'
                }.get(msg['role'], '❓')
                
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{role_emoji} [{timestamp}] {content}")
                
                if msg['tool_name']:
                    print(f"   🔧 Tool: {msg['tool_name']}")
                    
        except Exception as e:
            print(f"❌ Error retrieving history: {e}")
            logger.log_error("history_display_failed", str(e), "Failed to display conversation history")
    
    def _show_recent_conversations(self):
        """Show recent conversations summary"""
        try:
            conversations = self.conversation_history.get_recent_conversations(limit=10)
            
            if not conversations:
                print("📝 No conversations found.")
                return
                
            print(f"📝 Recent Conversations ({len(conversations)}):")
            print("-" * 50)
            
            for conv in conversations:
                started = conv['started_at'][:19] if conv['started_at'] else 'Unknown'
                status = "🟢 Active" if not conv['ended_at'] else "⚪ Ended"
                message_count = conv['message_count'] or 0
                
                print(f"ID: {conv['id']} | {status} | {started}")
                print(f"   📄 {conv['title']}")
                print(f"   💬 {message_count} messages")
                print()
                
        except Exception as e:
            print(f"❌ Error retrieving conversations: {e}")
            logger.log_error("conversations_display_failed", str(e), "Failed to display conversations")
    
    def _show_conversation_stats(self):
        """Show conversation analytics"""
        try:
            stats = self.conversation_history.get_conversation_analytics()
            
            if not stats:
                print("📊 No conversation statistics available.")
                return
                
            print("📊 Conversation Statistics:")
            print("-" * 30)
            print(f"Total Conversations: {stats.get('total_conversations', 0)}")
            print(f"Total Messages: {stats.get('total_messages', 0)}")
            print(f"Avg Messages/Conversation: {stats.get('avg_messages_per_conversation', 0):.1f}")
            print(f"Tool Messages: {stats.get('tool_messages', 0)}")
            print(f"Corrections: {stats.get('correction_messages', 0)}")
            print(f"Session ID: {stats.get('session_id', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error retrieving statistics: {e}")
            logger.log_error("stats_display_failed", str(e), "Failed to display conversation statistics")
    
    def _search_similar_messages(self):
        """Search for similar messages interactively"""
        try:
            search_query = input("🔍 Enter search query: ").strip()
            if not search_query:
                print("❌ Search query cannot be empty")
                return
            
            print(f"🔍 Searching for messages similar to: '{search_query}'")
            similar_messages = self.message_embeddings.search_similar_messages(
                query=search_query,
                k=10,
                exclude_conversation_ids=[self.conversation_history.current_conversation_id or 0],
                min_similarity_score=0.3
            )
            
            if not similar_messages:
                print("📝 No similar messages found")
                return
            
            print(f"📝 Found {len(similar_messages)} similar messages:")
            print("-" * 60)
            
            for i, msg in enumerate(similar_messages, 1):
                similarity = msg.get('similarity_score', 0)
                role_emoji = {'user': '👤', 'assistant': '🤖'}.get(msg['role'], '❓')
                conv_title = msg.get('conversation_title', 'Unknown')
                created_at = msg.get('created_at', 'Unknown')[:19]
                
                content = msg['original_content'][:150] + "..." if len(msg['original_content']) > 150 else msg['original_content']
                
                print(f"{i}. {role_emoji} Similarity: {similarity:.3f} | Conv: {conv_title} | {created_at}")
                print(f"   {content}")
                print()
            
        except Exception as e:
            print(f"❌ Error searching messages: {e}")
            logger.log_error("message_search_failed", str(e), "Failed to search similar messages")
    
    def _show_embedding_stats(self):
        """Show message embedding statistics"""
        try:
            stats = self.message_embeddings.get_index_statistics()
            
            if 'error' in stats:
                print(f"❌ Error retrieving embedding stats: {stats['error']}")
                return
            
            print("🔍 Message Embedding Statistics:")
            print("-" * 40)
            print(f"Messages Indexed: {stats.get('total_messages_indexed', 0)}")
            print(f"Last Indexed ID: {stats.get('last_indexed_message_id', 0)}")
            print(f"Vector Dimension: {stats.get('index_dimension', 0)}")
            print(f"Embedding Model: {stats.get('embedding_model', 'Unknown')}")
            print(f"Persistence: {'Enabled' if stats.get('persistence_enabled') else 'Disabled'}")
            
            build_time = stats.get('index_build_time')
            if build_time:
                print(f"Index Build Time: {build_time:.2f}s")
            
            # Show message role distribution
            if 'message_roles' in stats:
                print(f"\nMessage Distribution:")
                for role, count in stats['message_roles'].items():
                    print(f"  {role}: {count}")
            
            conversations_count = stats.get('conversations_indexed', 0)
            if conversations_count > 0:
                print(f"\nConversations Indexed: {conversations_count}")
            
        except Exception as e:
            print(f"❌ Error retrieving embedding stats: {e}")
            logger.log_error("embedding_stats_failed", str(e), "Failed to display embedding statistics")
    
    def _rebuild_message_index(self):
        """Rebuild the message embedding index"""
        try:
            print("🔄 Rebuilding message embedding index...")
            confirmation = input("This will rebuild the entire index. Continue? (y/N): ").strip().lower()
            
            if confirmation not in ['y', 'yes']:
                print("❌ Index rebuild cancelled")
                return
            
            start_time = time.time()
            self.message_embeddings.rebuild_index()
            rebuild_time = time.time() - start_time
            
            # Get updated stats
            stats = self.message_embeddings.get_index_statistics()
            messages_count = stats.get('total_messages_indexed', 0)
            
            print(f"✅ Index rebuilt successfully!")
            print(f"   Messages indexed: {messages_count}")
            print(f"   Rebuild time: {rebuild_time:.2f}s")
            
            logger.log_system_event(
                "message_index_manual_rebuild",
                f"User initiated index rebuild: {messages_count} messages, {rebuild_time:.2f}s"
            )
            
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            logger.log_error("index_rebuild_failed", str(e), "Failed to rebuild message index")
    
    def _summarise_conversation(self, full_input: str = None):
        """Interactive conversation summarization"""
        try:
            if not self.model:
                print("❌ Model not available for summarization")
                return
            
            # Check if conversation ID was provided as parameter
            conv_id = None
            if full_input:
                parts = full_input.strip().split()
                if len(parts) >= 2:
                    try:
                        conv_id = int(parts[1])
                    except ValueError:
                        print(f"❌ Invalid conversation ID: '{parts[1]}'. Must be a number.")
                        return
            
            # Get recent conversations
            conversations = self.conversation_history.get_recent_conversations(limit=10)
            
            if not conversations:
                print("❌ No conversations found to summarize")
                return
            
            # If no ID provided, show interactive selection
            if conv_id is None:
                # Show recent conversations for user to choose from
                print("📝 Recent Conversations:")
                print("-" * 50)
                
                for i, conv in enumerate(conversations, 1):
                    started = conv['started_at'][:19] if conv['started_at'] else 'Unknown'
                    status = "🟢 Active" if not conv['ended_at'] else "⚪ Ended"
                    message_count = conv.get('message_count', 0)
                    title = conv.get('title', 'Untitled Conversation')
                    has_summary = "✅" if conv.get('summary') else "❌"
                    
                    print(f"{i:2d}. ID: {conv['id']} | {status} | {has_summary} Summary | {started}")
                    print(f"     📄 {title}")
                    print(f"     💬 {message_count} messages")
                    print()
                
                # Get user choice
                while True:
                    try:
                        choice = input("\n🔍 Enter conversation ID to summarize (or 'cancel'): ").strip()
                        
                        if choice.lower() in ['cancel', 'c', 'exit']:
                            print("❌ Summarization cancelled")
                            return
                        
                        conv_id = int(choice)
                        break
                        
                    except ValueError:
                        print("❌ Please enter a valid number or 'cancel'")
                        continue
            
            # Validate ID exists in the list
            valid_ids = [conv['id'] for conv in conversations]
            if conv_id not in valid_ids:
                print(f"❌ Invalid conversation ID: {conv_id}. Available IDs: {valid_ids}")
                return
            
            # Check if already summarized
            selected_conv = next(conv for conv in conversations if conv['id'] == conv_id)
            if selected_conv.get('summary'):
                print(f"ℹ️  Conversation already has a summary:")
                print(f"📄 Title: {selected_conv['title']}")
                print(f"📝 Summary: {selected_conv['summary'][:200]}...")
                
                overwrite = input("\n🔄 Regenerate summary? (y/N): ").strip().lower()
                if overwrite not in ['y', 'yes']:
                    print("❌ Summary generation cancelled")
                    return
            
            # Fetch conversation data
            print(f"📊 Fetching conversation data for ID {conv_id}...")
            conversation_data = self.conversation_history.get_conversation_for_summary(conv_id)
            
            if not conversation_data['messages']:
                print("❌ No messages found in this conversation")
                return
            
            message_count = len(conversation_data['messages'])
            tool_count = len(conversation_data['tools_used'])
            
            print(f"📈 Found {message_count} messages and {tool_count} tool uses")
            print("🧠 Generating AI summary...")
            
            # Generate summary using the model
            summary_result = self.model.summarize_conversation(conversation_data)
            
            # Display generated summary
            print("\n" + "="*60)
            print("🤖 GENERATED SUMMARY")
            print("="*60)
            print(f"📄 Title: {summary_result['title']}")
            print(f"\n📝 Summary:\n{summary_result['summary']}")
            print(f"\n🔧 Tool Usage:\n{summary_result['tool_usage_summary']}")
            print("="*60)
            
            # Ask user if they want to save
            save_choice = input("\n💾 Save this summary to the database? (Y/n): ").strip().lower()
            
            if save_choice in ['', 'y', 'yes']:
                # Save to database
                success = self.conversation_history.save_conversation_summary(
                    conv_id,
                    summary_result['title'],
                    summary_result['summary'],
                    summary_result['tool_usage_summary']
                )
                
                if success:
                    print("✅ Summary saved successfully!")
                    logger.log_system_event(
                        "conversation_summarized",
                        f"User summarized conversation {conv_id}: '{summary_result['title']}'"
                    )
                else:
                    print("❌ Failed to save summary to database")
            else:
                print("❌ Summary not saved")
            
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            logger.log_error("conversation_summarization_failed", str(e), "User-initiated conversation summarization failed")
    
    def _handle_exit(self):
        """Handle exit command and return exit status"""
        # End current conversation before exiting
        self.conversation_history.end_current_conversation("User ended session")
        print("Goodbye!")
        logger.log_system_event("session_ended", "User requested exit")
        return {'should_exit': True, 'reason': 'user_exit'}
