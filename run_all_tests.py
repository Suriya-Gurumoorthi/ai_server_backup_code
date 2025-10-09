#!/usr/bin/env python3
"""
Comprehensive test runner for all audio integration tests.
This script runs all tests in sequence and provides a complete report.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """Runs all audio integration tests"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    async def run_test_script(self, script_name: str, description: str) -> dict:
        """Run a test script and capture results"""
        logger.info(f"🧪 Running {description}...")
        logger.info("=" * 60)
        
        try:
            # Import and run the test
            if script_name == "test_server_quick":
                from test_server_quick import quick_test
                success = await quick_test()
                return {"success": success, "description": description}
            
            elif script_name == "test_audio_integration":
                from test_audio_integration import AudioIntegrationTester
                tester = AudioIntegrationTester()
                results = await tester.run_comprehensive_test()
                return {"success": results.get("overall_success", False), "description": description, "details": results}
            
            elif script_name == "test_vicidial_bridge":
                from test_vicidial_bridge import ViciDialBridgeTester
                tester = ViciDialBridgeTester()
                results = await tester.run_comprehensive_test()
                return {"success": results.get("overall_success", False), "description": description, "details": results}
            
            elif script_name == "test_bridge_config":
                from test_bridge_config import run_comprehensive_bridge_test
                results = await run_comprehensive_bridge_test()
                return {"success": results.get("overall_success", False), "description": description, "details": results}
            
            else:
                logger.error(f"Unknown test script: {script_name}")
                return {"success": False, "description": description, "error": "Unknown test script"}
                
        except Exception as e:
            logger.error(f"❌ Test {description} failed with exception: {e}")
            return {"success": False, "description": description, "error": str(e)}
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        logger.info("🚀 Starting comprehensive audio integration test suite...")
        logger.info(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Define test sequence
        tests = [
            ("test_bridge_config", "Bridge Configuration Test"),
            ("test_server_quick", "Quick Server Test"),
            ("test_audio_integration", "Audio Integration Test"),
            ("test_vicidial_bridge", "ViciDial Bridge Test")
        ]
        
        # Run each test
        for script_name, description in tests:
            try:
                result = await self.run_test_script(script_name, description)
                self.test_results[script_name] = result
                
                if result["success"]:
                    logger.info(f"✅ {description} PASSED")
                else:
                    logger.error(f"❌ {description} FAILED")
                    if "error" in result:
                        logger.error(f"   Error: {result['error']}")
                
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"💥 Test {description} crashed: {e}")
                self.test_results[script_name] = {
                    "success": False, 
                    "description": description, 
                    "error": f"Test crashed: {str(e)}"
                }
        
        # Generate final report
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate final test report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        logger.info("📊 FINAL TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️ Total duration: {duration:.2f} seconds")
        logger.info("=" * 80)
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"📈 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info("=" * 80)
        
        # Detailed results
        logger.info("📋 Detailed Results:")
        for script_name, result in self.test_results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            logger.info(f"   {result['description']}: {status}")
            
            if not result["success"] and "error" in result:
                logger.info(f"      Error: {result['error']}")
            
            if "details" in result and isinstance(result["details"], dict):
                details = result["details"]
                for key, value in details.items():
                    if key != "overall_success":
                        status = "✅" if value else "❌"
                        logger.info(f"      {key}: {status}")
        
        logger.info("=" * 80)
        
        # Overall result
        overall_success = all(result["success"] for result in self.test_results.values())
        
        if overall_success:
            logger.info("🎉 ALL TESTS PASSED! Audio integration is working correctly.")
            logger.info("✅ Your changes are working as expected.")
            logger.info("✅ Server is sending audio directly to ViciDial bridge.")
            logger.info("✅ ViciDial bridge is handling direct audio responses.")
            return 0
        else:
            logger.error("💥 SOME TESTS FAILED!")
            logger.error("❌ Please check the failed tests above.")
            logger.error("❌ Review server logs and configuration.")
            return 1

async def main():
    """Main test runner function"""
    runner = TestRunner()
    exit_code = await runner.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())
