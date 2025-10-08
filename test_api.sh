#!/bin/bash

# Test script for Ultravox AI Model API
# Make sure the server is running on port 8000 before executing these tests

BASE_URL="http://localhost:8000"

echo "=== Testing Ultravox AI Model API ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -X GET "$BASE_URL/health" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Test 2: Root Endpoint
echo "2. Testing Root Endpoint..."
curl -X GET "$BASE_URL/" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Test 3: Create a test audio file using sox (if available) or skip
echo "3. Creating test audio file..."
if command -v sox &> /dev/null; then
    # Create a 2-second sine wave at 440Hz using sox
    sox -n -r 16000 -b 16 test_audio.wav synth 2 sine 440 vol 0.3
    echo "Test audio file created: test_audio.wav"
elif command -v ffmpeg &> /dev/null; then
    # Create a 2-second sine wave using ffmpeg
    ffmpeg -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 -y test_audio.wav
    echo "Test audio file created: test_audio.wav"
else
    echo "Warning: Neither sox nor ffmpeg found. Creating a dummy audio file..."
    # Create a minimal WAV file header (44 bytes) + some data
    printf "RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x3e\x00\x00\x80\x7c\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00" > test_audio.wav
    # Add some silence data (2 seconds at 16kHz, 16-bit)
    dd if=/dev/zero bs=32000 count=1 >> test_audio.wav 2>/dev/null
    echo "Dummy audio file created: test_audio.wav"
fi

# Test 4: Process Audio File Upload
echo "4. Testing Audio File Upload..."
curl -X POST "$BASE_URL/process_audio_file" \
  -F "audio_file=@test_audio.wav" \
  -F "conversation_id=test_conversation_1" \
  -F "max_new_tokens=50" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Test 5: Get Conversation History
echo "5. Testing Get Conversation History..."
curl -X GET "$BASE_URL/conversation/test_conversation_1" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Test 6: Process Audio with Base64 (using the same test file)
echo "6. Testing Base64 Audio Processing..."
# Convert audio to base64
AUDIO_BASE64=$(base64 -w 0 test_audio.wav)

curl -X POST "$BASE_URL/process_audio" \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_base64\": \"$AUDIO_BASE64\",
    \"turns\": [
      {
        \"role\": \"system\",
        \"content\": \"You are a helpful AI assistant.\"
      },
      {
        \"role\": \"user\",
        \"content\": \"Hello, can you hear me?\"
      }
    ],
    \"max_new_tokens\": 30
  }" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Test 7: Clear Conversation
echo "7. Testing Clear Conversation..."
curl -X DELETE "$BASE_URL/conversation/test_conversation_1" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Test 8: Verify Conversation is Cleared
echo "8. Verifying Conversation is Cleared..."
curl -X GET "$BASE_URL/conversation/test_conversation_1" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s
echo ""

# Cleanup
echo "9. Cleaning up test files..."
rm -f test_audio.wav
echo "Test audio file removed."

echo ""
echo "=== All tests completed ==="
