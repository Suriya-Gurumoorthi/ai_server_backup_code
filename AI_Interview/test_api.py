#!/usr/bin/env python3
"""
Test script to verify the API is working correctly.
"""

import requests
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing AI Interview Evaluation API")
    print("="*50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Web interface
    print("\n2. Testing web interface...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Web interface accessible")
            print(f"   Content length: {len(response.text)} characters")
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False
    
    # Test 3: API documentation
    print("\n3. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
        else:
            print(f"❌ API documentation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API documentation error: {e}")
    
    print("\n" + "="*50)
    print("🎉 API Tests Completed!")
    print("="*50)
    print("📱 Web Interface: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("\n✅ Your API is ready to use!")
    
    return True

if __name__ == "__main__":
    test_api() 