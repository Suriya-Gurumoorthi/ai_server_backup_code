# Audio Placeholder Management Fixes

## Problem Identified

The system was encountering the error:
```
ValueError: Text contains too many audio placeholders. (Expected 1 placeholders)
```

This occurred because the conversation history was accumulating multiple `<|audio|>` placeholders across turns, but Ultravox expects exactly one.

## Root Cause Analysis

### Why Multiple Placeholders Occurred:
1. **Conversation History Accumulation**: Each turn was adding `<|audio|>` placeholders
2. **Legacy Function Logic**: `ensure_single_audio_placeholder()` was adding placeholders to ALL user turns
3. **System Prompt Pollution**: System prompts could contain placeholders from templates
4. **No Validation**: No defensive checks before inference

### The Ultravox Requirement:
- **Exactly 1** `<|audio|>` placeholder across ALL conversation turns
- **Must be** in the last user turn
- **No exceptions** - any other configuration causes the pipeline to fail

## Fixes Applied

### 1. **New Placeholder Management Logic**
```python
def ensure_one_audio_placeholder_last_user(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Removes all audio placeholders from all turns except the last user turn,
    and ensures only one <|audio|> at the end.
    """
    # Find last user turn
    last_user_idx = -1
    for idx in reversed(range(len(turns))):
        if turns[idx].get('role') == 'user':
            last_user_idx = idx
            break

    # Clean all turns
    for idx, turn in enumerate(turns):
        content = sanitize_audio_placeholders(turn.get("content", ""))
        
        if role == "user" and idx == last_user_idx:
            # Only the last user turn gets the audio marker
            content = f"{content}\n<|audio|>" if content else "<|audio|>"
        
        cleaned.append({"role": role, "content": content})
    
    return cleaned
```

### 2. **Defensive Validation**
```python
def validate_audio_placeholders(turns: List[Dict[str, str]]) -> bool:
    """Defensive check to ensure exactly one audio placeholder"""
    placeholder_count = sum('<|audio|>' in turn.get('content', '') for turn in turns)
    
    if placeholder_count == 0:
        logger.warning("No audio placeholders found")
        return False
    elif placeholder_count > 1:
        logger.error(f"Too many audio placeholders: {placeholder_count}")
        return False
    else:
        logger.info("Audio placeholder validation passed")
        return True
```

### 3. **System Prompt Sanitization**
```python
def build_system_prompt_with_context(user_name: str = None, additional_context: str = None) -> str:
    # ... build prompt ...
    
    # Sanitize any audio placeholders from the system prompt
    enhanced_prompt = sanitize_audio_placeholders(enhanced_prompt)
    
    return enhanced_prompt
```

### 4. **Pre-Inference Validation**
```python
# Ensure proper audio placeholder handling
conversation_turns = ensure_one_audio_placeholder_last_user(conversation_turns)

# Defensive check before inference
if not validate_audio_placeholders(conversation_turns):
    logger.error("Audio placeholder validation failed, attempting to fix...")
    conversation_turns = ensure_one_audio_placeholder_last_user(conversation_turns)
    if not validate_audio_placeholders(conversation_turns):
        return "Error: Audio placeholder validation failed"
```

## Key Changes Made

### 1. **Replaced Problematic Function**
- **Before**: `ensure_single_audio_placeholder()` - added placeholders to ALL user turns
- **After**: `ensure_one_audio_placeholder_last_user()` - only last user turn gets placeholder

### 2. **Added Validation Layer**
- **Before**: No validation before inference
- **After**: `validate_audio_placeholders()` with automatic retry logic

### 3. **Sanitized System Prompts**
- **Before**: System prompts could contain placeholders
- **After**: All system prompts are sanitized before use

### 4. **Defensive Programming**
- **Before**: Assumed placeholder logic was correct
- **After**: Validate and fix before every inference

## Expected Results

### Before (Broken):
```
Turn 1: {"role": "user", "content": "Hello <|audio|>"}
Turn 2: {"role": "assistant", "content": "Hi there!"}
Turn 3: {"role": "user", "content": "My name is John <|audio|>"}
# Result: 2 placeholders → ERROR
```

### After (Fixed):
```
Turn 1: {"role": "user", "content": "Hello"}
Turn 2: {"role": "assistant", "content": "Hi there!"}
Turn 3: {"role": "user", "content": "My name is John <|audio|>"}
# Result: 1 placeholder → SUCCESS
```

## Architecture Flow

```
Conversation History → Clean All Placeholders → Add to Last User Turn → Validate → Inference
```

### Detailed Steps:
1. **Clean History**: Remove all `<|audio|>` from all turns
2. **Find Last User**: Identify the most recent user turn
3. **Add Placeholder**: Add `<|audio|>` only to last user turn
4. **Validate**: Ensure exactly 1 placeholder exists
5. **Retry if Needed**: If validation fails, clean and retry
6. **Inference**: Proceed with validated conversation turns

## Benefits

1. **Eliminates Placeholder Errors**: No more "too many placeholders" errors
2. **Robust Validation**: Defensive checks prevent pipeline failures
3. **Automatic Recovery**: Retry logic handles edge cases
4. **Clean History**: Conversation history stays clean and manageable
5. **Ultravox Compliance**: Meets exact requirements of the Ultravox pipeline

## Testing

The system should now:
- Handle multi-turn conversations without placeholder errors
- Maintain clean conversation history
- Automatically fix placeholder issues
- Provide clear error messages if unfixable issues occur
- Work reliably with the Ultravox pipeline

## Configuration

- **Validation**: Enabled before every inference
- **Retry Logic**: One automatic retry attempt
- **Error Handling**: Graceful failure with clear messages
- **Logging**: Detailed logging for debugging

