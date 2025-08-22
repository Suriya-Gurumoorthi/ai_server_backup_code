#!/usr/bin/env python3
"""
Test script to verify remote access to the AI Interview API.
Run this from your PC to test connectivity to the server.
"""

import requests
import sys
import time

def test_remote_access(server_ip="10.80.2.40", port=8000):
    """Test remote access to the API server"""
    base_url = f"http://{server_ip}:{port}"
    
    print("🌐 Testing Remote Access to AI Interview API")
    print("="*50)
    print(f"Server: {server_ip}:{port}")
    print("="*50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Timestamp: {data['timestamp']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused. Server might not be accessible from your PC.")
        print("   Try using SSH tunnel or check network connectivity.")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout. Server might be slow or unreachable.")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Web interface
    print("\n2. Testing web interface...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Web interface accessible")
            print(f"   Content length: {len(response.text)} characters")
            if "AI Interview Evaluation" in response.text:
                print("   ✅ Correct page content detected")
            else:
                print("   ⚠️  Page content might be different than expected")
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False
    
    # Test 3: API documentation
    print("\n3. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
        else:
            print(f"❌ API documentation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API documentation error: {e}")
    
    print("\n" + "="*50)
    print("🎉 Remote Access Tests Completed!")
    print("="*50)
    print("📱 Web Interface URLs to try:")
    print(f"   • http://{server_ip}:{port}")
    print(f"   • http://{server_ip}:{port}/docs")
    print(f"   • http://{server_ip}:{port}/health")
    print("\n✅ Your API is accessible from your PC!")
    print("\n💡 If you still can't access it in Chrome:")
    print("   1. Try the URLs above directly")
    print("   2. Check if your PC and server are on the same network")
    print("   3. Try using SSH tunnel method")
    
    return True

def main():
    """Main function"""
    # You can change the server IP here if needed
    server_ip = "10.80.2.40"
    port = 8000
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        server_ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print(f"Testing connection to {server_ip}:{port}")
    print("Press Ctrl+C to stop\n")
    
    try:
        test_remote_access(server_ip, port)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 