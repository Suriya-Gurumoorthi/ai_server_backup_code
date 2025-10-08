#!/usr/bin/env python3
"""
Model Client - Use the Ultravox model via HTTP server (no reloading needed)
"""

import requests
import json
import time

class UltravoxClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
    
    def _make_request(self, action, **kwargs):
        """Make request to model server"""
        try:
            data = {'action': action, **kwargs}
            response = requests.post(self.server_url, json=data, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': f"Server error: {e}"}
        except json.JSONDecodeError:
            return {'error': "Invalid JSON response from server"}
    
    def chat(self, message, audio_file=None, system_prompt=None, max_tokens=50):
        """Chat with the model"""
        return self._make_request('chat', 
                                message=message, 
                                audio_file=audio_file,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens)
    
    def transcribe(self, audio_file, system_prompt=None):
        """Transcribe audio"""
        return self._make_request('transcribe', 
                                audio_file=audio_file,
                                system_prompt=system_prompt)
    
    def answer_question(self, question, audio_file=None, system_prompt=None):
        """Answer a question"""
        return self._make_request('answer', 
                                question=question,
                                audio_file=audio_file,
                                system_prompt=system_prompt)
    
    def creative_response(self, prompt, audio_file=None, system_prompt=None):
        """Generate creative response"""
        return self._make_request('creative', 
                                prompt=prompt,
                                audio_file=audio_file,
                                system_prompt=system_prompt)
    
    def get_status(self):
        """Get server status"""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {'error': 'Server not running'}

def main():
    print("🎯 Ultravox Model Client")
    print("=" * 40)
    
    client = UltravoxClient()
    
    # Check server status
    status = client.get_status()
    if 'error' in status:
        print("❌ Server not running. Please start the server first:")
        print("   python model_server.py")
        return
    
    print(f"✅ Server status: {status}")
    
    # Example usage
    print("\n💬 Example 1: Simple chat")
    result = client.chat("Hello! What can you help me with?")
    if 'error' not in result:
        print(f"🤖 Assistant: {result.get('response', 'No response')}")
        print(f"⏱️  Time: {result.get('inference_time', 0):.2f}s")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n❓ Example 2: Answer a question")
    result = client.answer_question("What is artificial intelligence?")
    if 'error' not in result:
        print(f"🤖 Assistant: {result.get('response', 'No response')}")
        print(f"⏱️  Time: {result.get('inference_time', 0):.2f}s")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n🎨 Example 3: Creative response")
    result = client.creative_response("Write a short poem about technology")
    if 'error' not in result:
        print(f"🤖 Assistant: {result.get('response', 'No response')}")
        print(f"⏱️  Time: {result.get('inference_time', 0):.2f}s")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main()
