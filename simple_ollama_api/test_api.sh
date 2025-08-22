#!/bin/bash

# Simple Ollama API Test Script
echo "🧪 Testing Simple Ollama API on 10.80.2.40:8080"
echo "================================================"

# Variables
API_URL="http://10.80.2.40:8080"
API_KEY="simple_ollama_key_2024"

echo ""
echo "1️⃣ Testing Health Check..."
curl -s "$API_URL/health" | jq '.'

echo ""
echo "2️⃣ Testing Valid Chat Request..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"prompt": "Hello! Say hi in 3 words."}' | jq '.'

echo ""
echo "3️⃣ Testing Complex Question..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"prompt": "What is 2+2?"}' | jq '.'

echo ""
echo "4️⃣ Testing Missing API Key (should fail)..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello! Say hi in 3 words."}' | jq '.'

echo ""
echo "5️⃣ Testing Wrong API Key (should fail)..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: wrong_key" \
  -d '{"prompt": "Hello! Say hi in 3 words."}' | jq '.'

echo ""
echo "✅ Testing complete!"
