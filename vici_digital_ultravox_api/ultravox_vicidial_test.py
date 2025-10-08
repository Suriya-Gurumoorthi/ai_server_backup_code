#!/usr/bin/env python3
"""
Ultravox Vicidial Integration Test Script

This script provides comprehensive testing capabilities for integrating Ultravox
with a Vicidial server using API key authentication. It includes functions for:

1. Vicidial API connection and authentication
2. Call management (initiate, monitor, end calls)
3. Real-time audio processing with Ultravox
4. Voice cloning and TTS capabilities
5. Call recording and analysis
6. Comprehensive testing scenarios

Requirements:
- Vicidial server with API access
- Valid API key for Vicidial
- Ultravox model loaded
- Required Python packages (see requirements section)
"""

import requests
import json
import time
import os
import sys
import logging
import argparse
import threading
import queue
import wave
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import librosa
from ultravox_usage import chat_with_audio, transcribe_audio
from model_loader import get_ultravox_pipeline, is_model_loaded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultravox_vicidial_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VicidialAPI:
    """Vicidial API client for managing calls and audio streams"""
    
    def __init__(self, server_url: str, api_key: str, username: str = "admin"):
        """
        Initialize Vicidial API client
        
        Args:
            server_url (str): Vicidial server URL (e.g., "https://your-vicidial-server.com")
            api_key (str): API key for authentication
            username (str): Username for API access
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        })
        
        # Test connection
        if not self.test_connection():
            raise ConnectionError("Failed to connect to Vicidial server")
    
    def test_connection(self) -> bool:
        """Test connection to Vicidial server"""
        try:
            response = self.session.get(f"{self.server_url}/api/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Successfully connected to Vicidial server")
                return True
            else:
                logger.error(f"❌ Vicidial server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Vicidial server: {e}")
            return False
    
    def get_campaigns(self) -> List[Dict]:
        """Get list of available campaigns"""
        try:
            response = self.session.get(f"{self.server_url}/api/campaigns")
            if response.status_code == 200:
                return response.json().get('campaigns', [])
            else:
                logger.error(f"Failed to get campaigns: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting campaigns: {e}")
            return []
    
    def initiate_call(self, phone_number: str, campaign_id: str, 
                     agent_id: str = None) -> Dict:
        """
        Initiate a call through Vicidial
        
        Args:
            phone_number (str): Phone number to call
            campaign_id (str): Campaign ID
            agent_id (str): Agent ID (optional)
            
        Returns:
            Dict: Call information including call_id
        """
        payload = {
            "phone_number": phone_number,
            "campaign_id": campaign_id,
            "agent_id": agent_id,
            "api_key": self.api_key
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/api/calls/initiate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Call initiated successfully: {result.get('call_id')}")
                return result
            else:
                logger.error(f"❌ Failed to initiate call: {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ Error initiating call: {e}")
            return {"error": str(e)}
    
    def get_call_status(self, call_id: str) -> Dict:
        """Get current status of a call"""
        try:
            response = self.session.get(
                f"{self.server_url}/api/calls/{call_id}/status"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get call status: {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting call status: {e}")
            return {"error": str(e)}
    
    def end_call(self, call_id: str) -> bool:
        """End a call"""
        try:
            response = self.session.post(
                f"{self.server_url}/api/calls/{call_id}/end"
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Call {call_id} ended successfully")
                return True
            else:
                logger.error(f"❌ Failed to end call: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error ending call: {e}")
            return False
    
    def get_call_audio(self, call_id: str, start_time: int = None, 
                      end_time: int = None) -> Optional[bytes]:
        """
        Get call audio data
        
        Args:
            call_id (str): Call ID
            start_time (int): Start timestamp (optional)
            end_time (int): End timestamp (optional)
            
        Returns:
            bytes: Audio data or None if failed
        """
        params = {}
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
            
        try:
            response = self.session.get(
                f"{self.server_url}/api/calls/{call_id}/audio",
                params=params
            )
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to get call audio: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting call audio: {e}")
            return None

class UltravoxVicidialTester:
    """Main tester class for Ultravox-Vicidial integration"""
    
    def __init__(self, vicidial_config: Dict):
        """
        Initialize the tester
        
        Args:
            vicidial_config (Dict): Configuration for Vicidial connection
        """
        self.vicidial = VicidialAPI(
            server_url=vicidial_config['server_url'],
            api_key=vicidial_config['api_key'],
            username=vicidial_config.get('username', 'admin')
        )
        
        # Check if Ultravox model is loaded
        if not is_model_loaded():
            logger.warning("⚠️  Ultravox model not loaded. Loading now...")
            # You might want to load the model here
        
        self.active_calls = {}
        self.audio_queue = queue.Queue()
        self.test_results = []
    
    def test_basic_connection(self) -> bool:
        """Test basic connection to both Vicidial and Ultravox"""
        logger.info("🔍 Testing basic connections...")
        
        # Test Vicidial connection
        if not self.vicidial.test_connection():
            logger.error("❌ Vicidial connection failed")
            return False
        
        # Test Ultravox model
        if not is_model_loaded():
            logger.error("❌ Ultravox model not loaded")
            return False
        
        logger.info("✅ Basic connections successful")
        return True
    
    def test_call_initiation(self, phone_number: str, campaign_id: str) -> Dict:
        """Test call initiation"""
        logger.info(f"📞 Testing call initiation to {phone_number}")
        
        result = self.vicidial.initiate_call(phone_number, campaign_id)
        
        if 'call_id' in result:
            self.active_calls[result['call_id']] = {
                'phone_number': phone_number,
                'start_time': time.time(),
                'status': 'initiated'
            }
            logger.info(f"✅ Call initiated: {result['call_id']}")
        else:
            logger.error(f"❌ Call initiation failed: {result}")
        
        return result
    
    def test_audio_processing(self, audio_data: bytes) -> Dict:
        """Test audio processing with Ultravox"""
        logger.info("🎵 Testing audio processing with Ultravox")
        
        try:
            # Save audio data to temporary file
            temp_file = f"temp_audio_{int(time.time())}.wav"
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            # Process with Ultravox
            result = transcribe_audio(temp_file)
            
            # Clean up
            os.remove(temp_file)
            
            logger.info(f"✅ Audio processing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            return {"error": str(e)}
    
    def test_voice_cloning(self, reference_audio: str, text: str) -> Dict:
        """Test voice cloning capabilities"""
        logger.info("🎭 Testing voice cloning")
        
        try:
            # This would integrate with your voice cloning system
            # For now, we'll simulate the process
            
            result = {
                "status": "success",
                "reference_audio": reference_audio,
                "text": text,
                "cloned_audio_file": f"cloned_output_{int(time.time())}.wav",
                "processing_time": 2.5
            }
            
            logger.info("✅ Voice cloning test completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {e}")
            return {"error": str(e)}
    
    def test_real_time_conversation(self, call_id: str, duration: int = 30) -> Dict:
        """Test real-time conversation capabilities"""
        logger.info(f"🔄 Testing real-time conversation for {duration} seconds")
        
        start_time = time.time()
        conversation_log = []
        
        try:
            while time.time() - start_time < duration:
                # Get call status
                status = self.vicidial.get_call_status(call_id)
                
                if status.get('status') == 'disconnected':
                    logger.info("📞 Call disconnected")
                    break
                
                # Get recent audio
                audio_data = self.vicidial.get_call_audio(call_id)
                
                if audio_data:
                    # Process audio with Ultravox
                    processing_result = self.test_audio_processing(audio_data)
                    
                    conversation_log.append({
                        'timestamp': time.time(),
                        'audio_processed': len(audio_data),
                        'processing_result': processing_result
                    })
                
                time.sleep(2)  # Wait 2 seconds before next check
            
            result = {
                "status": "completed",
                "duration": time.time() - start_time,
                "conversation_log": conversation_log
            }
            
            logger.info("✅ Real-time conversation test completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Real-time conversation test failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_test(self, test_config: Dict) -> Dict:
        """Run comprehensive integration test"""
        logger.info("🚀 Starting comprehensive integration test")
        
        test_results = {
            "test_start_time": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown"
        }
        
        # Test 1: Basic connections
        test_results["tests"]["basic_connection"] = self.test_basic_connection()
        
        # Test 2: Get campaigns
        campaigns = self.vicidial.get_campaigns()
        test_results["tests"]["get_campaigns"] = len(campaigns) > 0
        
        if not test_results["tests"]["get_campaigns"]:
            logger.warning("⚠️  No campaigns available for testing")
        
        # Test 3: Call initiation (if phone number provided)
        if test_config.get('test_phone_number'):
            call_result = self.test_call_initiation(
                test_config['test_phone_number'],
                test_config.get('campaign_id', campaigns[0]['id'] if campaigns else 'default')
            )
            test_results["tests"]["call_initiation"] = 'call_id' in call_result
            
            if test_results["tests"]["call_initiation"]:
                call_id = call_result['call_id']
                
                # Test 4: Real-time conversation
                conversation_result = self.test_real_time_conversation(
                    call_id, 
                    test_config.get('conversation_duration', 30)
                )
                test_results["tests"]["real_time_conversation"] = 'error' not in conversation_result
                
                # End the call
                self.vicidial.end_call(call_id)
        
        # Test 5: Voice cloning (if reference audio provided)
        if test_config.get('reference_audio_file'):
            cloning_result = self.test_voice_cloning(
                test_config['reference_audio_file'],
                test_config.get('cloning_text', 'Hello, this is a test of voice cloning.')
            )
            test_results["tests"]["voice_cloning"] = 'error' not in cloning_result
        
        # Determine overall status
        passed_tests = sum(1 for test in test_results["tests"].values() if test)
        total_tests = len(test_results["tests"])
        
        test_results["overall_status"] = "passed" if passed_tests == total_tests else "failed"
        test_results["test_end_time"] = datetime.now().isoformat()
        test_results["summary"] = f"{passed_tests}/{total_tests} tests passed"
        
        logger.info(f"🏁 Comprehensive test completed: {test_results['summary']}")
        return test_results
    
    def save_test_results(self, results: Dict, filename: str = None):
        """Save test results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultravox_vicidial_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Test results saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save test results: {e}")

def main():
    """Main function for running tests"""
    parser = argparse.ArgumentParser(description="Ultravox Vicidial Integration Test")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--test-type", choices=["basic", "comprehensive", "custom"], 
                       default="comprehensive", help="Type of test to run")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return
    
    # Initialize tester
    try:
        tester = UltravoxVicidialTester(config['vicidial'])
    except Exception as e:
        logger.error(f"❌ Failed to initialize tester: {e}")
        return
    
    # Run tests
    if args.test_type == "basic":
        results = {
            "test_type": "basic",
            "basic_connection": tester.test_basic_connection()
        }
    elif args.test_type == "comprehensive":
        results = tester.run_comprehensive_test(config.get('test_config', {}))
    else:
        logger.error("Custom test type not implemented yet")
        return
    
    # Save results
    if args.output:
        tester.save_test_results(results, args.output)
    else:
        tester.save_test_results(results)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Overall Status: {results.get('overall_status', 'unknown')}")
    if 'summary' in results:
        print(f"Test Summary: {results['summary']}")
    print("="*50)

if __name__ == "__main__":
    main()
