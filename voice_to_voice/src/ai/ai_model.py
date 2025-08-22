import os
from src.utils.logger import setup_logger
from config.settings import AI_MODEL_SETTINGS
from config.roles import get_role

class AIModel:
    # Global memory store to persist across requests
    _global_memory = {}
    
    def __init__(self):
        self.logger = setup_logger()
        self.model = None
        self.memory = None
        self.current_role = "assistant"
        self.initialize_model()
        self.initialize_memory()
    
    def set_role(self, role_name):
        """Set the current role for the AI model."""
        self.current_role = role_name
        self.logger.info(f"AI model role set to: {role_name}")
    
    def get_current_role(self):
        """Get the current role name."""
        return self.current_role
    
    def initialize_memory(self):
        """Initialize conversation memory for context retention."""
        try:
            from langchain.memory import ConversationBufferMemory
            
            # Use global memory if available, otherwise create new
            if not hasattr(self, '_session_id'):
                self._session_id = 'default'
            
            if self._session_id not in self._global_memory:
                self._global_memory[self._session_id] = ConversationBufferMemory(
                    memory_key="history",
                    return_messages=True,
                    input_key="input"
                )
                self.logger.info(f"Created new global memory for session: {self._session_id}")
            
            self.memory = self._global_memory[self._session_id]
            self.logger.info(f"Using existing global memory for session: {self._session_id}")
            
        except ImportError:
            self.logger.warning("LangChain memory not available, using simple list memory")
            if self._session_id not in self._global_memory:
                self._global_memory[self._session_id] = []
            self.memory = self._global_memory[self._session_id]
        except Exception as e:
            self.logger.error(f"Failed to initialize memory: {e}")
            if self._session_id not in self._global_memory:
                self._global_memory[self._session_id] = []
            self.memory = self._global_memory[self._session_id]
    
    def set_session_id(self, session_id):
        """Set the session ID for memory management."""
        self._session_id = session_id
        self.initialize_memory()
    
    def get_session_id(self):
        """Get the current session ID."""
        return getattr(self, '_session_id', 'default')
    
    def initialize_model(self):
        """Initialize the AI model for generating responses."""
        try:
            # Try to use Ollama if available
            try:
                from langchain_community.llms import Ollama
                self.model = Ollama(
                    model="llama3.2:3b",  # Use a smaller, faster model
                    temperature=AI_MODEL_SETTINGS['temperature']
                )
                self.logger.info("Initialized Ollama LLM model")
                return
            except ImportError:
                self.logger.warning("Ollama not available, trying Ultravox with Llama")
            
            # Try to use Ultravox with Llama (the original model)
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                # Use the original Ultravox model with Llama
                model_name = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
                
                # Set pad token
                tokenizer.pad_token = tokenizer.eos_token
                
                self.model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU
                    trust_remote_code=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                self.logger.info("Initialized Ultravox with Llama model")
                return
            except Exception as e:
                self.logger.warning(f"Ultravox with Llama not available: {e}")
            
            # Fallback: Try DialoGPT
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                model_name = "microsoft/DialoGPT-small"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                tokenizer.pad_token = tokenizer.eos_token
                
                self.model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,
                    pad_token_id=tokenizer.eos_token_id
                )
                self.logger.info("Initialized DialoGPT-small model")
                return
            except Exception as e:
                self.logger.warning(f"DialoGPT not available: {e}")
            
            # Final fallback: Use rule-based system
            self.model = "rule_based"
            self.logger.info("Using rule-based response system")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI model: {e}")
            self.model = "rule_based"
    
    def generate_response(self, user_input, role_context="assistant", conversation_history=None):
        """Generate a response based on user input and context."""
        try:
            if self.model == "rule_based":
                return self._rule_based_response(user_input, role_context)
            
            elif hasattr(self.model, 'invoke'):  # Ollama model
                # Use LangChain conversation chain with memory
                return self._generate_with_memory(user_input, role_context)
            
            elif hasattr(self.model, '__call__'):  # Transformers pipeline
                try:
                    # For Ultravox/Llama models, use a simpler prompt
                    if "ultravox" in str(self.model).lower() or "llama" in str(self.model).lower():
                        # Ultravox/Llama specific prompt
                        prompt = f"User: {user_input}\nAssistant:"
                    else:
                        # For other models, use the full prompt
                        prompt = self._build_prompt(user_input, role_context, conversation_history)
                    
                    result = self.model(prompt, max_length=200, num_return_sequences=1, do_sample=True)
                    response = result[0]['generated_text']
                    
                    # Extract only the new response part
                    if prompt in response:
                        response = response.replace(prompt, "").strip()
                    
                    if response and len(response) > 10:
                        # Update memory with this interaction
                        self._update_memory(user_input, response)
                        return response
                    else:
                        # Fall back to rule-based system if model gives poor response
                        self.logger.warning(f"Model gave poor response: '{response}', using rule-based fallback")
                        return self._rule_based_response(user_input, role_context)
                        
                except Exception as e:
                    self.logger.error(f"Error in model generation: {e}")
                    return self._rule_based_response(user_input, role_context)
            
            else:
                return self._rule_based_response(user_input, role_context)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._rule_based_response(user_input, role_context)
    
    def _generate_with_memory(self, user_input, role_context):
        """Generate response using LangChain with memory."""
        try:
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            # Debug: Check memory state
            memory_summary = self.get_memory_summary()
            self.logger.info(f"Memory state before generation: {memory_summary}")
            
            # Get role-specific prompt from configuration
            role_config = get_role(self.current_role)
            role_prompt = role_config.get("prompt", "You are a helpful AI assistant.")
            
            # Check if this is a new conversation (no history or very short history)
            is_new_conversation = False
            if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory:
                # Check if we have very few messages (indicating new conversation)
                messages = self.memory.chat_memory.messages
                is_new_conversation = len(messages) <= 2  # Only 1-2 messages means new conversation
            else:
                is_new_conversation = True
            
            # For ongoing conversations, use a more focused prompt without repeating the full role context
            if is_new_conversation:
                # New conversation - include full role context
                template = f"""{role_prompt}

Current conversation:
{{history}}

Human: {{input}}
Assistant:"""
            else:
                # Ongoing conversation - use focused prompt
                if self.current_role == "fahad_sales":
                    template = """You are Fahad from Novel Office. Continue the conversation naturally, focusing on providing helpful information about office spaces, services, and addressing the user's specific needs. Be professional, friendly, and avoid repeating introductions.

Current conversation:
{history}

Human: {input}
Assistant:"""
                else:
                    # For other roles, use a generic focused prompt
                    template = """Continue the conversation naturally, focusing on the user's needs and providing helpful responses.

Current conversation:
{history}

Human: {input}
Assistant:"""

            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )

            # Create a chain with memory
            chain = LLMChain(
                llm=self.model,
                prompt=prompt,
                memory=self.memory,
                verbose=False
            )

            # Generate response
            response = chain.run(input=user_input)
            
            # Update memory with this interaction
            self._update_memory(user_input, response.strip())
            
            # Debug: Check memory state after generation
            memory_summary_after = self.get_memory_summary()
            self.logger.info(f"Memory state after generation: {memory_summary_after}")
            
            self.logger.info(f"Generated response with memory for '{user_input}': {response}")
            return response.strip()

        except Exception as e:
            self.logger.error(f"Error in memory-based generation: {e}")
            # Fallback to simple prompt without memory
            role_config = get_role(self.current_role)
            role_prompt = role_config.get("prompt", "You are a helpful AI assistant.")
            prompt = f"{role_prompt}\n\nHuman: {user_input}\nAssistant:"
            response = self.model.invoke(prompt)
            self._update_memory(user_input, response.strip())
            return response.strip()
    
    def _update_memory(self, user_input, response):
        """Update conversation memory with the new interaction."""
        try:
            if hasattr(self.memory, 'save_context'):
                # LangChain memory
                self.memory.save_context(
                    {"input": user_input},
                    {"output": response}
                )
            elif isinstance(self.memory, list):
                # Simple list memory
                self.memory.append({"user": user_input, "assistant": response})
                # Keep only last 10 interactions to prevent memory overflow
                if len(self.memory) > 10:
                    self.memory = self.memory[-10:]
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
    
    def clear_memory(self):
        """Clear conversation memory."""
        try:
            if hasattr(self.memory, 'clear'):
                self.memory.clear()
            elif isinstance(self.memory, list):
                self.memory.clear()
            self.logger.info("Conversation memory cleared")
        except Exception as e:
            self.logger.error(f"Error clearing memory: {e}")
    
    def get_memory_summary(self):
        """Get a summary of the conversation memory."""
        try:
            if hasattr(self.memory, 'chat_memory'):
                return f"Memory has {len(self.memory.chat_memory.messages)} messages"
            elif hasattr(self.memory, 'buffer'):
                return f"Memory has {len(self.memory.buffer)} messages"
            elif isinstance(self.memory, list):
                return f"Memory has {len(self.memory)} interactions"
            else:
                return "Memory not available"
        except Exception as e:
            self.logger.error(f"Error getting memory summary: {e}")
            return "Memory error"
    
    def _build_prompt(self, user_input, role_context, conversation_history=None):
        """Build a prompt for the AI model."""
        role_prompts = {
            "assistant": "You are a helpful AI assistant. Respond naturally and concisely to user queries.",
            "guide": "You are a knowledgeable guide. Provide detailed, informative responses to help users learn and understand topics.",
            "friend": "You are a friendly conversational partner. Be warm, supportive, and engaging in your responses.",
            "expert": "You are an expert in your field. Provide accurate, detailed, and professional responses."
        }
        
        role_prompt = role_prompts.get(role_context, role_prompts["assistant"])
        
        # Build conversation context
        context = f"{role_prompt}\n\n"
        
        if conversation_history:
            for turn in conversation_history[-5:]:  # Last 5 turns
                context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        
        context += f"User: {user_input}\nAssistant:"
        
        return context
    
    def _rule_based_response(self, user_input, role_context):
        """Generate a simple rule-based response when AI models are not available."""
        user_input_lower = user_input.lower()
        
        # Enhanced response patterns with more specific answers
        responses = {
            "assistant": {
                "greeting": ["Hello! How can I help you today?", "Hi there! What can I assist you with?", "Greetings! How may I be of service?"],
                "name": ["I'm an AI assistant, here to help you with your questions and tasks.", "You can call me Assistant. I'm here to help!"],
                "chatgpt": ["ChatGPT is an AI language model developed by OpenAI. It can help with writing, coding, answering questions, and having conversations. It's trained on a large dataset and can understand and respond to human language naturally."],
                "ai": ["Artificial Intelligence (AI) is technology that enables computers to perform tasks that typically require human intelligence, like understanding language, recognizing images, and making decisions."],
                "weather": ["I don't have access to real-time weather data, but I can help you find weather information online."],
                "time": ["I don't have access to real-time clock data, but you can check your device's clock."],
                "help": ["I'm here to help! You can ask me questions, have conversations, or get assistance with various topics."],
                "thanks": ["You're welcome! Is there anything else I can help you with?", "Happy to help! Let me know if you need anything else."],
                "default": ["That's an interesting question! I'm here to help you with information and conversation.", "I'd be happy to discuss that with you. What specific aspect would you like to explore?"]
            },
            "guide": {
                "greeting": ["Welcome! I'm here to guide you through various topics and help you learn.", "Hello! I'm your knowledge guide. What would you like to explore today?"],
                "name": ["I'm your Knowledge Guide, here to help you learn and explore various topics.", "You can call me Guide. I'm here to provide detailed information and explanations."],
                "chatgpt": ["ChatGPT is an advanced AI language model created by OpenAI. It uses deep learning to understand and generate human-like text. It can assist with writing, programming, problem-solving, and educational content. The model is trained on diverse internet text and can engage in meaningful conversations."],
                "ai": ["Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that normally require human intelligence. This includes machine learning, natural language processing, computer vision, and robotics."],
                "help": ["I'm your knowledge guide! I can help you learn about various topics, provide detailed explanations, and guide you through complex subjects."],
                "thanks": ["You're very welcome! I'm glad I could help guide you. What else would you like to learn about?"],
                "default": ["That's a great topic to explore! Let me provide you with some detailed information about that.", "I'd be happy to guide you through that subject. What specific aspect would you like to focus on?"]
            }
        }
        
        role_responses = responses.get(role_context, responses["assistant"])
        
        # Check for specific patterns
        if any(word in user_input_lower for word in ["hello", "hi", "hey", "greetings"]):
            return self._random_choice(role_responses["greeting"])
        elif any(word in user_input_lower for word in ["name", "who are you", "what are you"]):
            return self._random_choice(role_responses["name"])
        elif any(word in user_input_lower for word in ["chatgpt", "gpt", "openai", "who is chatgpt", "what is chatgpt"]):
            return self._random_choice(role_responses["chatgpt"])
        elif any(word in user_input_lower for word in ["ai", "artificial intelligence", "machine learning"]):
            return self._random_choice(role_responses["ai"])
        elif any(word in user_input_lower for word in ["microsoft", "bill gates", "satya nadella"]):
            return "Microsoft is a technology company founded by Bill Gates and Paul Allen. It's currently led by CEO Satya Nadella and is known for products like Windows, Office, and Azure cloud services."
        elif any(word in user_input_lower for word in ["google", "alphabet", "sundar pichai"]):
            return "Google is a technology company that's part of Alphabet Inc. It's led by CEO Sundar Pichai and is known for search, Android, Chrome, and various AI technologies."
        elif any(word in user_input_lower for word in ["deepseek", "deep seek"]):
            return "DeepSeek is an AI research company that develops large language models and AI technologies. They're known for their advanced AI models and research in artificial intelligence."
        elif any(word in user_input_lower for word in ["weather", "temperature"]):
            return self._random_choice(role_responses["weather"])
        elif any(word in user_input_lower for word in ["time", "clock", "hour"]):
            return self._random_choice(role_responses["time"])
        elif any(word in user_input_lower for word in ["help", "assist", "support"]):
            return self._random_choice(role_responses["help"])
        elif any(word in user_input_lower for word in ["thank", "thanks", "appreciate"]):
            return self._random_choice(role_responses["thanks"])
        else:
            # For unknown topics, try to give a more helpful response
            return f"I understand you're asking about '{user_input}'. Let me try to help you with that. Could you provide more specific details about what you'd like to know?"
    
    def _random_choice(self, options):
        """Choose a random option from a list."""
        import random
        return random.choice(options) 