from llama_cpp import Llama
import os, json, sys, time, logging
from terminal.animations import Animations
from utils.conversation_logger import conversation_logger
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class Phi3Model:
    def __init__(self, model_path=None, max_context_tokens=3500):
        self.llm = None
        self.is_loaded = False
        self.is_healthy = True
        self.last_health_check = 0
        self.health_check_interval = 1800  # 30 minutes
        self.consecutive_failures = 0
        self.max_failures = 3
        self.animator = Animations()
        self.max_context_tokens = max_context_tokens
        
        # Make model path configurable
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "Phi-3-mini-4k-instruct-q4.gguf"
        )
        
        # Initialize tokenizer for token counting
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer, close approximation
            except:
                self.tokenizer = None
        else:
            self.tokenizer = None


    
    def load_model(self):
        if self.is_loaded:
            return

        def load():
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=8,
                verbose=False
            )
            self.is_loaded = True

        self.animator.run_with_animation(load, message="Loading Phi3 Model...")
    
    def health_check(self):
        """Check model health with simple inference test"""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_healthy
        
        try:
            if not self.is_loaded:
                self.is_healthy = False
                conversation_logger.log_health_check(False, "Model not loaded")
                return False
            
            # Simple test inference
            test_response = self.llm(
                "<|user|>\nHello<|end|>\n<|assistant|>",
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response and test_response.get("choices"):
                self.consecutive_failures = 0
                self.is_healthy = True
                conversation_logger.log_health_check(True, "Test inference successful")
                logger.info("Model health check passed")
            else:
                raise Exception("Empty response from model")
                
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Model health check failed: {e}")
            conversation_logger.log_health_check(False, f"Test failed: {str(e)}")
            
            if self.consecutive_failures >= self.max_failures:
                self.is_healthy = False
                logger.error("Model marked as unhealthy after consecutive failures")
        
        self.last_health_check = current_time
        return self.is_healthy
    
    def count_tokens(self, text):
        """Count tokens in text. Falls back to word count * 1.3 if tiktoken unavailable."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 word ≈ 1.3 tokens
            return int(len(text.split()) * 1.3)
    
    def _handle_generation_error(self, e, attempt, max_attempts, operation="generation"):
        """Common error handling for generation methods"""
        self.consecutive_failures += 1
        logger.warning(f"{operation.capitalize()} attempt {attempt + 1} failed: {e}")
        conversation_logger.log_retry_attempt(attempt + 1, max_attempts, str(e))
        
        if attempt < max_attempts - 1:
            time.sleep(1)  # Brief pause before retry
        else:
            # Final attempt failed
            if self.consecutive_failures >= self.max_failures:
                self.is_healthy = False
            conversation_logger.log_error(f"{operation}_failed", str(e), f"After {max_attempts} attempts")
            raise RuntimeError(f"{operation.capitalize()} failed after {max_attempts} attempts: {e}")
    
    def generate(self,system_prompt, prompt, max_tokens=512, temperature=0.7, retries=1):
        """Generate response for evaluation tasks (non-streaming)"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.health_check():
            raise RuntimeError("Model is unhealthy. Please restart the application.")
            
        formatted_prompt = f"<|system|>{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        # Log the prompt being sent
        conversation_logger.log_model_prompt("evaluation", prompt)
        
        start_time = time.time()
        
        for attempt in range(retries + 1):
            try:
                response = self.llm(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|end|>"]
                )
                
                if response and response.get("choices") and response["choices"][0].get("text"):
                    self.consecutive_failures = 0
                    response_text = response["choices"][0]["text"]
                    response_time = time.time() - start_time
                    
                    # Log successful response
                    conversation_logger.log_model_response(
                        "evaluation", 
                        response_text, 
                        streaming=False, 
                        tokens=self.count_tokens(response_text)
                    )
                    conversation_logger.log_system_event("generation_complete", f"Response time: {response_time:.2f}s")
                    
                    return response_text
                else:
                    raise Exception("Empty or invalid response")
                    
            except Exception as e:
                self._handle_generation_error(e, attempt, retries + 1, "generation")
    
    def generate_streaming(self, prompt, max_tokens=512, temperature=0.7):
        """Generate response with real-time streaming for chat interactions"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.health_check():
            raise RuntimeError("Model is unhealthy. Please restart the application.")
            
        
        # Log the prompt being sent for streaming
        conversation_logger.log_model_prompt("chat_streaming", prompt)
        
        try:
            # Create streaming generator
            stream = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|end|>"],
                stream=True  # Enable streaming
            )
            
            full_response = ""
            print("Assistant: ", end="", flush=True)  # Print prefix
            
            for output in stream:
                if output['choices'][0]['finish_reason'] is None:
                    token = output['choices'][0]['text']
                    full_response += token
                    print(token, end="", flush=True)  # Stream each token
            
            print()  # New line after streaming complete
            self.consecutive_failures = 0
            
            # Log the complete streamed response
            conversation_logger.log_model_response(
                "chat_streaming", 
                full_response, 
                streaming=True, 
                tokens=self.count_tokens(full_response)
            )
            
            return full_response
            
        except Exception as e:
            self._handle_generation_error(e, 0, 1, "streaming")
    
    def generate_with_context(self, user_input, embedding_context=None, tool_context=None, max_tokens=512, temperature=0.7):
        """Generate response with proper prompt engineering and context"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Build the prompt based on available context
        prompt = self._build_contextual_prompt(
            user_input, embedding_context, tool_context
        )
        
        return self.generate_streaming(prompt, max_tokens, temperature)
    
    def _build_contextual_prompt(self, user_input, embed_context=None, tool_context=None):
        """Build context-aware prompt (model-specific logic)"""
        prompt_parts = []
        
        # Add embedding context if available
        if embed_context:
            prompt_parts.append("You are an assistant that answers user queries using available information. Use the information provided to respond naturally and directly. Do not mention the source of the information, do not refer to past conversations, and do not add disclaimers. Only provide clear, concise, natural answers.")
            prompt_parts.append("<|system|>\n**Available Information:**")
            for i, context in enumerate(embed_context[:3], 1):  # Top 3 similar
                prompt_parts.append(f"{i}. User: {context['user_message']}")
                prompt_parts.append(f"   Assistant: {context['assistant_response']}")
            prompt_parts.append("<|end|>\n")
        
        # Add tool context if available
        if tool_context:
            prompt_parts.append("You are an assistant that answers user queries using available internal data. Use the provided information to respond naturally and directly. Do not mention how you got the information, do not mention any tools, and do not add disclaimers. Only provide a natural, concise answer.")
            prompt_parts.append("<|system|>\n**Available Information:**")
            prompt_parts.append(f"Tool used: {tool_context['name']}")
            prompt_parts.append(f"Result: {tool_context['result']}")
            prompt_parts.append("<|end|>\n")

        prompt_parts.append(f"<|user|>\n {user_input}<|end|>\n<|assistant|>")
        return "\n".join(prompt_parts)



    def model_evaluate_tool_selection_with_confidence(self, user_query, tool):
        """Evaluate tool selection with confidence scoring and uncertainty handling"""
        def animate():
            start_time = time.time()
            
            prompt = f"""
            You are an expert in tool relevance evaluation.
            User query: "{user_query}"

            Candidate tool:
            - Name: {tool['name']}
            - Description: {tool['query_examples']}
            - Function: {tool['python_function']}

            Evaluate if this tool should be used to answer the user's query.
            
            Respond with a JSON object containing:
            - "decision": "true" or "false" 
            - "confidence": a number from 0.0 to 1.0 indicating your certainty
            - "reasoning": brief explanation of your decision
            
            Example: {{"decision": "true", "confidence": 0.9, "reasoning": "Tool directly matches user's request"}}
            """

            raw_output = self.generate(prompt, max_tokens=200).strip()
            evaluation_time = time.time() - start_time
            
            # Try to parse JSON response first
            try:
                result = json.loads(raw_output)
                
                decision = result.get("decision", "").lower() == "true"
                confidence = float(result.get("confidence", 0.5))
                reasoning = result.get("reasoning", "No reasoning provided")
                
                # Validate confidence range
                confidence = max(0.0, min(1.0, confidence))
                
                evaluation_result = {
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "uncertainty": 1.0 - confidence
                }
                
                # Log the tool evaluation with all details
                conversation_logger.log_tool_evaluation(
                    user_query, 
                    tool['name'], 
                    decision, 
                    confidence, 
                    reasoning, 
                    evaluation_time
                )
                
                return evaluation_result
                
            except (json.JSONDecodeError, ValueError, KeyError):
                # JSON parsing failed - default to negative (conservative approach)
                evaluation_result = {"decision": False, "confidence": 0.1, 
                       "reasoning": "Failed to parse JSON response", "uncertainty": 0.9}
                
                # Log the failed evaluation
                conversation_logger.log_tool_evaluation(
                    user_query, 
                    tool['name'], 
                    False, 
                    0.1, 
                    "Failed to parse JSON response", 
                    evaluation_time
                )
                conversation_logger.log_error("evaluation_parse_failed", f"Raw output: {raw_output}")
                
                return evaluation_result
        
        return self.animator.run_with_animation(animate, message=f"Evaluating tool usage for {tool['name']}...")

    def summarize_conversation(self, conversation_data):
        """
        Generate comprehensive conversation summary with title and tool usage
        
        Args:
            conversation_data: Dictionary with messages and tool usage
            
        Returns:
            Dictionary with title, summary, and tool_usage_summary
        """
        try:
            messages = conversation_data.get('messages', [])
            tools_used = conversation_data.get('tools_used', [])
            
            if not messages:
                return {
                    'title': 'Empty Conversation',
                    'summary': 'No messages found in this conversation.',
                    'tool_usage_summary': 'No tools were used.'
                }
            
            # Build conversation text for summarization
            conversation_text = []
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content'][:500]  # Limit length for processing
                conversation_text.append(f"{role}: {content}")
            
            full_conversation = "\n".join(conversation_text)
            
            # Determine conversation size and create appropriate prompt
            message_count = len(messages)
            if message_count <= 30:
                summary_system_prompt = (
                "You are an assistant that summarizes conversations."
                "Use the provided conversation information to generate a concise, single-sentence summary of the main intent and topic."
                "Do not include punctuation, labels, emojis, formatting, or disclaimers."
                "Do not mention tools or how information was obtained."
                "Only output the summary sentence."
            )
            else:
                summary_system_prompt = (
                "Analyze this long conversation and provide a detailed summary "
                "(2-3 paragraphs) highlighting key topics, important decisions, "
                "and any unresolved questions. Output only the summary, without "
                "punctuation, labels, or disclaimers."
            )
            
            summary = self.generate(summary_system_prompt, full_conversation, max_tokens=300, temperature=0.5)
            
            # Generate conversation title
            title_system_prompt = f"""You are an assistant that summarizes conversations. Use the provided information to generate a single concise title. Do not mention tools, sources, or past conversations. Output only the title as instructed, without labels, punctuation, or disclaimers."""
            title_prompt = full_conversation[:1000]
            
            title = self.generate(title_system_prompt, title_prompt, max_tokens=30, temperature=0.3).strip()
            # Clean up title (remove quotes, extra text, and common AI additions)
            title = title.replace('"', '').replace("'", "").strip()
            
            # Remove "Title:" if it appears (common AI response pattern)
            if title.lower().startswith('title:'):
                title = title[6:].strip()
            if title.lower().endswith(' title:'):
                title = title[:-7].strip()
            
            # Remove everything after opening parentheses (common AI explanations)
            if '(' in title:
                title = title.split('(')[0].strip()
            
            # Remove everything after "Note:" or similar
            if 'note:' in title.lower():
                title = title.split('note:')[0].strip()
            
            # Remove trailing colons or periods
            title = title.rstrip(':.')
            
            # Limit to 5 words max as requested
            words = title.split()
            if len(words) > 5:
                title = ' '.join(words[:5])
            
            # Final length check and fallback
            if len(title) > 50:
                title = title[:47] + "..."
            
            # Fallback if title is empty after cleanup
            if not title.strip():
                title = "Conversation Summary"
            
            # Generate tool usage summary
            tool_summary = "No tools were used."
            if tools_used:
                tools_id = []
                for tool in tools_used:
                    tools_id.append(tool.get('name', 'Unknown'))

            if tools_id:
                tool_summary = ', '.join(tools_id)
            
            return {
                'title': title,
                'summary': summary.strip(),
                'tool_usage_summary': tool_summary.strip()
            }
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return {
                'title': 'Conversation Summary',
                'summary': 'Failed to generate conversation summary.',
                'tool_usage_summary': 'Failed to analyze tool usage.'
            }