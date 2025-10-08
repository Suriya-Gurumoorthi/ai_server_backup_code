#!/usr/bin/env python3
"""
Quick Vicidial Integration Test

A simplified script for quick testing of Vicidial API integration with Ultravox.
This script provides basic functionality without the full comprehensive testing suite.
"""

import requests
import json
import time
import os
from datetime import datetime

class QuickVicidialTest:
    """Quick test class for Vicidial API integration"""
    
    def __init__(self, server_url: str, api_key: str):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        })
    
    def test_connection(self):
        """Test basic connection to Vicidial server"""
        print("🔍 Testing Vicidial connection...")
        try:
            response = self.session.get(f"{self.server_url}/api/health", timeout=10)
            if response.status_code == 200:
                print("✅ Vicidial connection successful")
                return True
            else:
                print(f"❌ Vicidial server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_campaigns(self):
        """Get available campaigns"""
        print("📋 Getting campaigns...")
        try:
            response = self.session.get(f"{self.server_url}/api/campaigns")
            if response.status_code == 200:
                campaigns = response.json().get('campaigns', [])
                print(f"✅ Found {len(campaigns)} campaigns")
                return campaigns
            else:
                print(f"❌ Failed to get campaigns: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error getting campaigns: {e}")
            return []
    
    def test_call_initiation(self, phone_number: str, campaign_id: str):
        """Test call initiation"""
        print(f"📞 Testing call initiation to {phone_number}...")
        
        payload = {
            "phone_number": phone_number,
            "campaign_id": campaign_id,
            "api_key": self.api_key
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/api/calls/initiate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                call_id = result.get('call_id')
                print(f"✅ Call initiated successfully: {call_id}")
                return call_id
            else:
                print(f"❌ Failed to initiate call: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error initiating call: {e}")
            return None
    
    def monitor_call(self, call_id: str, duration: int = 30):
        """Monitor a call for specified duration"""
        print(f"🔄 Monitoring call {call_id} for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                response = self.session.get(f"{self.server_url}/api/calls/{call_id}/status")
                if response.status_code == 200:
                    status = response.json()
                    print(f"📊 Call status: {status.get('status', 'unknown')}")
                    
                    if status.get('status') == 'disconnected':
                        print("📞 Call disconnected")
                        break
                else:
                    print(f"⚠️  Failed to get call status: {response.status_code}")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"⚠️  Error monitoring call: {e}")
                time.sleep(5)
        
        print("✅ Call monitoring completed")
    
    def end_call(self, call_id: str):
        """End a call"""
        print(f"📞 Ending call {call_id}...")
        try:
            response = self.session.post(f"{self.server_url}/api/calls/{call_id}/end")
            if response.status_code == 200:
                print("✅ Call ended successfully")
                return True
            else:
                print(f"❌ Failed to end call: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error ending call: {e}")
            return False

def main():
    """Main function for quick testing"""
    print("🚀 Quick Vicidial Integration Test")
    print("=" * 40)
    
    # Configuration - modify these values
    SERVER_URL = "https://your-vicidial-server.com"
    API_KEY = "your_api_key_here"
    TEST_PHONE = "+1234567890"
    
    # Initialize tester
    tester = QuickVicidialTest(SERVER_URL, API_KEY)
    
    # Test 1: Connection
    if not tester.test_connection():
        print("❌ Cannot proceed without connection")
        return
    
    # Test 2: Get campaigns
    campaigns = tester.get_campaigns()
    if not campaigns:
        print("⚠️  No campaigns available")
        return
    
    campaign_id = campaigns[0]['id']  # Use first available campaign
    print(f"📋 Using campaign: {campaign_id}")
    
    # Test 3: Initiate call
    call_id = tester.test_call_initiation(TEST_PHONE, campaign_id)
    if not call_id:
        print("❌ Cannot proceed without call initiation")
        return
    
    # Test 4: Monitor call
    tester.monitor_call(call_id, duration=30)
    
    # Test 5: End call
    tester.end_call(call_id)
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    main()
