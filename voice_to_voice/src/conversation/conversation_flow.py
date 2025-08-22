from src.conversation.prompt_manager import PromptManager
from src.ai.ai_model import AIModel
from src.utils.logger import setup_logger

class ConversationFlow:
    def __init__(self, role_name="assistant", session_id=None):
        self.logger = setup_logger()
        self.prompt_manager = PromptManager(role_name=role_name)
        self.ai_model = AIModel()
        
        # Set session ID for memory management
        if session_id:
            self.ai_model.set_session_id(session_id)
        else:
            # Use role-based session ID
            self.ai_model.set_session_id(f"{role_name}_session")
        
        # Set the AI model role
        self.ai_model.set_role(role_name)
        self.logger.info(f"Conversation flow initialized with role: {role_name}")

    def change_role(self, new_role_name):
        """Change the current role of the AI model."""
        self.ai_model.set_role(new_role_name)
        self.prompt_manager.role_name = new_role_name
        # Clear memory when changing roles to ensure clean transition
        self.clear_memory()
        self.logger.info(f"Role changed to: {new_role_name}")

    def generate_response(self, user_input, history):
        """Generate a response based on user input and history."""
        try:
            # Ensure the AI model is using the current role
            current_role = self.ai_model.get_current_role()
            if current_role != self.prompt_manager.role_name:
                self.ai_model.set_role(self.prompt_manager.role_name)
                self.logger.info(f"Updated AI model role to match prompt manager: {self.prompt_manager.role_name}")
            
            # Generate response using AI model with memory and role context
            response = self.ai_model.generate_response(
                user_input=user_input,
                role_context=self.prompt_manager.role_name
            )
            
            self.logger.info(f"Generated response for '{user_input}' with role '{self.ai_model.get_current_role()}': {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            # Fallback response
            return f"I apologize, but I'm having trouble processing that right now. Could you please rephrase your question?"
    
    def clear_memory(self):
        """Clear the conversation memory."""
        try:
            self.ai_model.clear_memory()
            self.logger.info("Conversation memory cleared")
        except Exception as e:
            self.logger.error(f"Error clearing memory: {e}")
    
    def get_memory_summary(self):
        """Get a summary of the conversation memory."""
        try:
            return self.ai_model.get_memory_summary()
        except Exception as e:
            self.logger.error(f"Error getting memory summary: {e}")
            return "Memory not available"
    
    def get_session_id(self):
        """Get the current session ID."""
        return self.ai_model.get_session_id()
    
    def get_current_role(self):
        """Get the current role name."""
        return self.ai_model.get_current_role()