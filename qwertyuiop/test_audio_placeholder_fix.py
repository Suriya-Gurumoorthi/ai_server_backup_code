#!/usr/bin/env python3
"""
Test script to verify the audio placeholder fix.
Tests the ensure_one_audio_placeholder_last_user function.
"""

import sys
import os

# Add the server directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ensure_one_audio_placeholder_last_user, sanitize_audio_placeholders

def test_sanitize_audio_placeholders():
    """Test the sanitize_audio_placeholders function."""
    print("Testing sanitize_audio_placeholders function...")
    
    test_cases = [
        ("Hello <|audio|> world", "Hello world"),
        ("<|audio|> Hello world", "Hello world"),
        ("Hello <|audio|> world <|audio|>", "Hello world"),
        ("<|audio|>", ""),
        ("", ""),
        ("Hello world", "Hello world"),
        ("Hello <|audio| world", "Hello world"),
        ("Hello |audio|> world", "Hello world"),
        ("Hello |audio| world", "Hello world"),
    ]
    
    for input_text, expected in test_cases:
        result = sanitize_audio_placeholders(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {status}: '{input_text}' -> '{result}' (expected: '{expected}')")
        if result != expected:
            return False
    
    return True

def test_ensure_one_audio_placeholder():
    """Test the ensure_one_audio_placeholder_last_user function."""
    print("\nTesting ensure_one_audio_placeholder_last_user function...")
    
    # Test case 1: Normal conversation
    turns1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    result1 = ensure_one_audio_placeholder_last_user(turns1)
    expected_last_user = "How are you? <|audio|>"
    
    print(f"  Test 1 - Normal conversation:")
    print(f"    Last user content: '{result1[-1]['content']}'")
    print(f"    Expected: '{expected_last_user}'")
    status1 = "✅ PASS" if result1[-1]['content'] == expected_last_user else "❌ FAIL"
    print(f"    {status1}")
    
    # Test case 2: Conversation with multiple audio placeholders
    turns2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello <|audio|>"},
        {"role": "assistant", "content": "Hi there! <|audio|>"},
        {"role": "user", "content": "How are you? <|audio|>"}
    ]
    
    result2 = ensure_one_audio_placeholder_last_user(turns2)
    
    print(f"\n  Test 2 - Multiple audio placeholders:")
    print(f"    System content: '{result2[0]['content']}'")
    print(f"    Assistant content: '{result2[2]['content']}'")
    print(f"    Last user content: '{result2[-1]['content']}'")
    
    # Check that all non-last turns have no audio placeholders
    system_clean = "<|audio|>" not in result2[0]['content']
    assistant_clean = "<|audio|>" not in result2[2]['content']
    last_user_has_one = result2[-1]['content'].count('<|audio|>') == 1
    
    status2 = "✅ PASS" if (system_clean and assistant_clean and last_user_has_one) else "❌ FAIL"
    print(f"    {status2}")
    
    # Test case 3: Empty conversation
    turns3 = []
    result3 = ensure_one_audio_placeholder_last_user(turns3)
    status3 = "✅ PASS" if result3 == [] else "❌ FAIL"
    print(f"\n  Test 3 - Empty conversation: {status3}")
    
    return status1 == "✅ PASS" and status2 == "✅ PASS" and status3 == "✅ PASS"

def main():
    """Run all tests."""
    print("🧪 Testing Audio Placeholder Fix")
    print("=" * 50)
    
    test1_passed = test_sanitize_audio_placeholders()
    test2_passed = test_ensure_one_audio_placeholder()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  sanitize_audio_placeholders: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  ensure_one_audio_placeholder: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! Audio placeholder fix is working correctly.")
        return 0
    else:
        print("\n💥 Some tests FAILED! Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

