#!/bin/bash

# Test script for audio processing
BASE_URL="http://localhost:8000"

echo "Testing audio processing with converted WAV file..."

# Convert base64 to a temporary file
AUDIO_BASE64=$(base64 -w 0 test_audio.wav)

# Create JSON payload
cat > test_payload.json << EOF
{
  "audio_base64": "$AUDIO_BASE64",
  "turns": [
    {
      "role": "system",
      "content": "You are a friendly and helpful character. You love to answer questions for people."
    }
  ],
  "max_new_tokens": 30
}
EOF

echo "Testing base64 audio processing..."
curl -X POST "$BASE_URL/process_audio" \
  -H "Content-Type: application/json" \
  -d @test_payload.json \
  -w "\nHTTP Status: %{http_code}\n"

echo ""
echo "Testing file upload with fresh conversation..."
curl -X POST "$BASE_URL/process_audio_file" \
  -F "audio_file=@test_audio.wav" \
  -F "conversation_id=fresh_test_$(date +%s)" \
  -F "max_new_tokens=30" \
  -w "\nHTTP Status: %{http_code}\n"

# Cleanup
rm -f test_payload.json
