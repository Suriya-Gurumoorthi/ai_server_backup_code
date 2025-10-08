# Conversation History and Context Management Fixes

## Problem Identified

The conversation logic was broken because the system was not maintaining proper conversation history per session. Key issues:

1. **No Per-Session Conversation History**: Always started fresh with `default_turns`
2. **No User Identity Tracking**: User names weren't extracted or stored
3. **Static System Prompts**: No dynamic context injection
4. **No Dialogue Memory**: Model couldn't remember previous exchanges

## Root Causes

- **Static `default_turns`**: Every conversation started from scratch
- **No Name Extraction**: User introductions were ignored
- **Missing Context Persistence**: No way to maintain conversation state
- **Poor Memory Integration**: RAG was only used for similarity search, not dialogue context

## Fixes Applied

### 1. **Per-Session Conversation History**
- **Before**: `conversation_turns: default_turns.copy()`
- **After**: `conversation_turns: []` (empty, grows over time)
- **Implementation**: Each session maintains its own evolving conversation history

### 2. **User Name Extraction and Storage**
```python
def extract_user_name(text: str) -> str:
    """Extract user name using regex patterns"""
    patterns = [
        r"my name is ([A-Za-z\s]+)",
        r"i'm ([A-Za-z\s]+)",
        r"i am ([A-Za-z\s]+)",
        # ... more patterns
    ]
```

- **Storage**: `connection["user_name"] = extracted_name`
- **Usage**: Injected into system prompt dynamically

### 3. **Dynamic System Prompt with Context**
```python
def build_system_prompt_with_context(user_name: str = None, additional_context: str = None) -> str:
    """Build system prompt with user context"""
    if user_name:
        context_parts.append(f"You are speaking to {user_name}.")
    if additional_context:
        context_parts.append(f"Additional context: {additional_context}")
```

- **User Context**: "You are speaking to Suriya."
- **Memory Context**: Retrieved from vector database
- **Dynamic Updates**: System prompt updates as conversation progresses

### 4. **Proper Conversation Turn Management**
```python
# Add user input to conversation
conversation_turns.append({"role": "user", "content": user_input})

# Generate response using full conversation history
result = await run_model_inference(conversation_turns)

# Add assistant response to conversation history
conversation_turns.append({"role": "assistant", "content": response_text})

# Update conversation history in connection state
connection["conversation_turns"] = conversation_turns
```

### 5. **Conversation History Trimming**
```python
def trim_conversation_history(turns: List[Dict[str, str]], max_turns: int = 20) -> List[Dict[str, str]]:
    """Trim conversation history to keep only the most recent turns"""
    # Keep system prompt + recent turns
    system_turn = turns[0] if turns and turns[0].get("role") == "system" else None
    recent_turns = turns[-(max_turns-1):] if system_turn else turns[-max_turns:]
```

- **Memory Management**: Prevents context window overflow
- **System Prompt Preservation**: Always keeps the first system turn
- **Recent Context**: Maintains last N conversation turns

## New Architecture Flow

```
Audio Input → Transcription → Name Extraction → Context Building → Conversation History Update → Response Generation → History Storage
```

### Detailed Flow:
1. **Transcribe Audio**: Convert to text
2. **Extract Name**: Look for name introduction patterns
3. **Update Context**: Build system prompt with user name and memory
4. **Add to History**: Append user input to conversation turns
5. **Generate Response**: Use full conversation history for context
6. **Store Response**: Add assistant response to conversation history
7. **Update State**: Save conversation history in connection state

## Key Functions Added/Modified

### New Functions:
- `extract_user_name()`: Regex-based name extraction
- `build_system_prompt_with_context()`: Dynamic prompt building
- `trim_conversation_history()`: Memory management

### Modified Functions:
- `generate_ai_response()`: Complete rewrite for proper conversation management
- Connection initialization: Added `user_name` and empty `conversation_turns`

## Expected Results

### Before (Broken):
```
User: "My name is Suriya"
Alexa: "My name is Alexa, not Suriya"
```

### After (Fixed):
```
User: "My name is Suriya"
Alexa: "Thank you, Suriya. It's nice to meet you. How can I help you today?"

User: "What's my name?"
Alexa: "Your name is Suriya. Now, let's discuss your background..."
```

## Benefits

1. **Natural Conversations**: Proper dialogue flow with context
2. **User Identity Recognition**: Names are remembered and used
3. **Contextual Responses**: AI responds based on conversation history
4. **Memory Integration**: RAG context enhances dialogue
5. **Scalable Architecture**: Per-session state management

## Testing

The system should now:
- Extract and remember user names
- Maintain conversation context across turns
- Generate contextually appropriate responses
- Use memory to enhance dialogue
- Handle long conversations with proper trimming

## Configuration

- **Max Conversation Turns**: 20 (configurable)
- **Name Extraction Patterns**: 7 regex patterns
- **Memory Context Length**: 300 characters
- **System Prompt Updates**: Dynamic based on user context

