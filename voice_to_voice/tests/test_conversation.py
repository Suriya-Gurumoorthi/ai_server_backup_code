import pytest
from src.conversation.conversation_flow import ConversationFlow
from src.core.voice_session import VoiceSession

def test_conversation_flow():
    flow = ConversationFlow()
    session = VoiceSession()
    response = flow.generate_response("Hello", session.get_history())
    assert isinstance(response, str)
    assert "Echo: Hello" in response