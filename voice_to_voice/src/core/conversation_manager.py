from src.stt.speech_recognizer import SpeechRecognizer
from src.tts.voice_generator import VoiceGenerator
from src.conversation.conversation_flow import ConversationFlow
from src.core.voice_session import VoiceSession
from src.utils.logger import setup_logger
from config.settings import STORAGE_SETTINGS, CONVERSATION_SETTINGS, TERMINATION_KEYWORDS

class ConversationManager:
    def __init__(self, role_name="assistant"):
        self.logger = setup_logger()
        self.session = VoiceSession()
        self.stt = SpeechRecognizer()
        self.tts = VoiceGenerator()
        self.conversation = ConversationFlow(role_name=role_name)
        self.max_turns = CONVERSATION_SETTINGS['max_turns']

    def start_conversation(self):
        self.logger.info("Starting conversation loop...")
        turn_count = 0
        try:
            while turn_count < self.max_turns:
                # Get audio input (from mic or file)
                audio_path = self.session.get_audio_input()
                if not audio_path:
                    self.logger.info("No audio input, exiting...")
                    break

                # Convert audio to text
                user_text = self.stt.recognize_speech(audio_path)
                if not user_text:
                    self.logger.warning("No speech recognized, continuing...")
                    continue

                user_text = user_text.lower()
                self.logger.info(f"User said: {user_text}")
                self.session.add_message("user", user_text)

                # Check for termination
                if any(keyword in user_text for keyword in TERMINATION_KEYWORDS):
                    self.logger.info("Termination keyword detected, ending conversation.")
                    break

                # Generate response
                response = self.conversation.generate_response(user_text, self.session.get_history())
                self.logger.info(f"Assistant response: {response}")
                self.session.add_message("assistant", response)

                # Convert response to audio
                output_audio = self.tts.generate_speech(response)
                self.logger.info(f"Generated audio: {output_audio}")

                turn_count += 1
        except KeyboardInterrupt:
            self.logger.info("Conversation interrupted by user.")
        except Exception as e:
            self.logger.error(f"Error in conversation: {e}")
            raise