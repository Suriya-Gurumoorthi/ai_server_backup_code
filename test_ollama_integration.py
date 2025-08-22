#!/usr/bin/env python3
"""
Simple test script to demonstrate Ollama integration
"""

import requests
import json

def test_ollama_api():
    """Test Ollama API directly"""
    print("=== Testing Ollama API ===")
    
    # Test basic generation
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.2:3b',
            'prompt': 'Explain what Ollama is in one sentence.'
        }
    )
    
    if response.status_code == 200:
        # Parse streaming response
        lines = response.text.strip().split('\n')
        full_response = ""
        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response += data['response']
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ API Response: {full_response}")
    else:
        print(f"❌ API Error: {response.status_code}")

def test_model_manager():
    """Test the model manager script"""
    print("\n=== Testing Model Manager ===")
    
    import subprocess
    try:
        # Test listing models
        result = subprocess.run(
            ['python', 'model_manager.py', 'list'],
            capture_output=True,
            text=True,
            cwd='/home/novel'
        )
        print(f"✅ Model List: {result.stdout.strip()}")
        
        # Test model testing
        result = subprocess.run(
            ['python', 'model_manager.py', 'test', 'llama3.2:3b', 'What is AI?'],
            capture_output=True,
            text=True,
            cwd='/home/novel'
        )
        print(f"✅ Model Test: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"❌ Model Manager Error: {e}")

def test_simple_chat():
    """Test simple chat functionality"""
    print("\n=== Testing Simple Chat ===")
    
    try:
        from langchain_community.llms import Ollama
        
        # Create LLM instance
        llm = Ollama(model="llama3.2:3b", temperature=0.7)
        
        # Test simple question
        question = "What is the capital of France?"
        print(f"Question: {question}")
        
        response = llm.invoke(question)
        print(f"✅ Response: {response}")
        
    except Exception as e:
        print(f"❌ LangChain Error: {e}")
        print("Note: This requires proper LangChain setup")

def main():
    print("🚀 Ollama Integration Test Suite\n")
    
    # Test 1: Direct API
    test_ollama_api()
    
    # Test 2: Model Manager
    test_model_manager()
    
    # Test 3: LangChain Integration
    test_simple_chat()
    
    print("\n🎉 Test Suite Complete!")
    print("\nTo use Ollama in your project:")
    print("1. Direct API: Use requests to http://localhost:11434/api/generate")
    print("2. Command Line: Use ollama run llama3.2:3b")
    print("3. Python Scripts: Use the model_manager.py or llm_example.py")
    print("4. LangChain: Use langchain_community.llms.Ollama")

if __name__ == "__main__":
    main()
