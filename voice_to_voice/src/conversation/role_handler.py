from src.utils.logger import setup_logger
from config.roles import ROLES

class RoleHandler:
    def __init__(self):
        self.logger = setup_logger()
        self.roles = ROLES

    def get_role_prompt(self, role_name):
        """Get the prompt for a specific role."""
        role = self.roles.get(role_name, {})
        prompt = role.get("prompt", "")
        self.logger.info(f"Retrieved prompt for role '{role_name}': {prompt}")
        return prompt