import argparse
from src.core.conversation_manager import ConversationManager
from src.utils.logger import setup_logger
from config.roles import list_available_roles, get_role
from config.settings import CONVERSATION_SETTINGS, WEB_SETTINGS
from src.web.app import create_app

def run_cli(role_name, input_file=None):
    """Run the voice-to-voice system in CLI mode."""
    logger = setup_logger()
    logger.info("Starting voice-to-voice system in CLI mode...")

    # Validate role
    available_roles = list_available_roles()
    if role_name not in available_roles:
        logger.warning(f"Role '{role_name}' not found. Using default 'assistant' role.")
        role_name = "assistant"

    # Initialize conversation manager
    try:
        conv_manager = ConversationManager(role_name=role_name)
        conv_manager.start_conversation()
    except Exception as e:
        logger.error(f"Error running conversation: {e}")
        raise

def run_web():
    """Run the voice-to-voice system in web mode."""
    logger = setup_logger()
    logger.info("Starting voice-to-voice system in web mode...")
    app = create_app()
    app.run(
        host=WEB_SETTINGS["host"],
        port=WEB_SETTINGS["port"],
        debug=WEB_SETTINGS["debug"]
    )

def main():
    parser = argparse.ArgumentParser(description="Voice-to-Voice AI System")
    parser.add_argument("--role", default="assistant", help="AI role to use (e.g., assistant, guide)")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="Run in CLI or web mode")
    parser.add_argument("--input", default=None, help="Path to input audio file (optional)")
    args = parser.parse_args()

    if args.mode == "web":
        run_web()
    else:
        run_cli(args.role, args.input)

if __name__ == "__main__":
    main()