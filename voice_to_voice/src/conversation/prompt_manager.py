from src.conversation.role_handler import RoleHandler
from src.utils.logger import setup_logger
from config.roles import get_role

class PromptManager:
    def __init__(self, role_name="assistant"):
        self.logger = setup_logger()
        self.role_name = role_name
        self.role_handler = RoleHandler()

    def build_prompt(self, user_input, history):
        """Build a prompt for the AI based on role and history."""
        role = get_role(self.role_name)
        role_prompt = role.get("prompt", "")
        # Include history in prompt
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        prompt = f"{role_prompt}\n{history_text}\nUser: {user_input}"
        self.logger.info(f"Built prompt: {prompt}")
        return prompt