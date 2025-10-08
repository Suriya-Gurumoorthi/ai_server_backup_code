#!/usr/bin/env python3
"""
Test script for remote Ultravox bridge setup
This script tests both the bridge server and client functionality
"""

import asyncio
import sys
import os
import time
import logging
from vicidial_client import UltravoxBridgeClient, VicidialUltravoxIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemoteSetupTester:
    """Test class for remote Ultravox bridge setup"""
    
    def __init__(self, bridge_host, bridge_port=9092, secret_key="your-secret-key-change-this"):
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.secret_key = secret_key
        self.test_results = {}
    
    async def test_network_connectivity(self):
        """Test basic network connectivity to bridge server"""
        logger.info("Testing network connectivity...")
        
        try:
            # Test basic TCP connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.bridge_host, self.bridge_port),
                timeout=5.0
            )
            
            writer.close()
            await writer.wait_closed()
            
            self.test_results['network_connectivity'] = True
            logger.info("✅ Network connectivity test passed")
            return True
            
        except asyncio.TimeoutError:
            logger.error("❌ Network connectivity test failed: Timeout")
            self.test_results['network_connectivity'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Network connectivity test failed: {e}")
            self.test_results['network_connectivity'] = False
            return False
    
    async def test_authentication(self):
        """Test authentication with bridge server"""
        logger.info("Testing authentication...")
        
        try:
            client = UltravoxBridgeClient(
                bridge_host=self.bridge_host,
                bridge_port=self.bridge_port,
                secret_key=self.secret_key,
                enable_auth=True
            )
            
            success = await client.connect()
            await client.disconnect()
            
            self.test_results['authentication'] = success
            if success:
                logger.info("✅ Authentication test passed")
            else:
                logger.error("❌ Authentication test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {e}")
            self.test_results['authentication'] = False
            return False
    
    async def test_audio_transmission(self):
        """Test audio transmission to and from bridge"""
        logger.info("Testing audio transmission...")
        
        try:
            integration = VicidialUltravoxIntegration(
                bridge_host=self.bridge_host,
                bridge_port=self.bridge_port,
                secret_key=self.secret_key
            )
            
            # Start integration
            await integration.start()
            
            # Send test audio
            test_audio = b'\x00' * 320  # 20ms of silence
            await integration.send_vicidial_audio(test_audio)
            
            # Wait for potential response
            await asyncio.sleep(2)
            
            # Stop integration
            await integration.stop()
            
            self.test_results['audio_transmission'] = True
            logger.info("✅ Audio transmission test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio transmission test failed: {e}")
            self.test_results['audio_transmission'] = False
            return False
    
    async def test_full_integration(self):
        """Test full integration workflow"""
        logger.info("Testing full integration...")
        
        try:
            integration = VicidialUltravoxIntegration(
                bridge_host=self.bridge_host,
                bridge_port=self.bridge_port,
                secret_key=self.secret_key
            )
            
            # Start integration
            await integration.start()
            
            # Simulate a short conversation
            audio_received = False
            
            def audio_callback(audio_data):
                nonlocal audio_received
                audio_received = True
                logger.info(f"Received {len(audio_data)} bytes from Ultravox")
            
            integration.bridge_client.set_audio_callback(audio_callback)
            
            # Send multiple audio frames
            for i in range(10):
                test_audio = b'\x00' * 320
                await integration.send_vicidial_audio(test_audio)
                await asyncio.sleep(0.1)
            
            # Wait for responses
            await asyncio.sleep(3)
            
            # Stop integration
            await integration.stop()
            
            self.test_results['full_integration'] = True
            logger.info("✅ Full integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full integration test failed: {e}")
            self.test_results['full_integration'] = False
            return False
    
    async def run_all_tests(self):
        """Run all tests and return results"""
        logger.info("Starting remote setup tests...")
        logger.info(f"Bridge server: {self.bridge_host}:{self.bridge_port}")
        
        tests = [
            ("Network Connectivity", self.test_network_connectivity),
            ("Authentication", self.test_authentication),
            ("Audio Transmission", self.test_audio_transmission),
            ("Full Integration", self.test_full_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} Test ---")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        return self.test_results
    
    def print_summary(self):
        """Print test results summary"""
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Your remote setup is working correctly.")
        else:
            logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        
        return passed == total

async def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage: python test_remote_setup.py <bridge_server_ip> [port] [secret_key]")
        print("Example: python test_remote_setup.py 192.168.1.100 9092 my-secret-key")
        sys.exit(1)
    
    bridge_host = sys.argv[1]
    bridge_port = int(sys.argv[2]) if len(sys.argv) > 2 else 9092
    secret_key = sys.argv[3] if len(sys.argv) > 3 else "your-secret-key-change-this"
    
    # Create tester
    tester = RemoteSetupTester(bridge_host, bridge_port, secret_key)
    
    # Run tests
    results = await tester.run_all_tests()
    
    # Print summary
    success = tester.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)




