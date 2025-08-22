#!/usr/bin/env python3
"""
Example API client for AI Interview Evaluation System.
Demonstrates how to use the API programmatically.
"""

import requests
import time
import json
from pathlib import Path

class InterviewEvaluationClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def check_health(self):
        """Check API health and model status"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def upload_audio(self, audio_file_path):
        """Upload an audio file for evaluation"""
        try:
            with open(audio_file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/upload", files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Upload failed: {e}")
            return None
        except FileNotFoundError:
            print(f"Audio file not found: {audio_file_path}")
            return None
    
    def get_status(self, job_id):
        """Get processing status for a job"""
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Status check failed: {e}")
            return None
    
    def get_results(self, job_id):
        """Get evaluation results for a completed job"""
        try:
            response = requests.get(f"{self.base_url}/results/{job_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Results retrieval failed: {e}")
            return None
    
    def evaluate_audio(self, audio_file_path, max_wait_time=300):
        """Complete evaluation workflow"""
        print(f"🎤 Starting evaluation for: {audio_file_path}")
        
        # Check health first
        health = self.check_health()
        if not health:
            print("❌ API is not available")
            return None
        
        print(f"✅ API Health: {health['status']}")
        print(f"✅ Model Loaded: {health['model_loaded']}")
        
        # Upload audio file
        upload_result = self.upload_audio(audio_file_path)
        if not upload_result:
            print("❌ Upload failed")
            return None
        
        job_id = upload_result['job_id']
        print(f"✅ Upload successful. Job ID: {job_id}")
        
        # Monitor processing
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status = self.get_status(job_id)
            if not status:
                print("❌ Status check failed")
                return None
            
            print(f"📊 Status: {status['status']}")
            
            if status['status'] == 'completed':
                # Get results
                results = self.get_results(job_id)
                if results:
                    print("🎉 Evaluation completed!")
                    return results
                else:
                    print("❌ Failed to retrieve results")
                    return None
            
            elif status['status'] == 'failed':
                print(f"❌ Processing failed: {status.get('error', 'Unknown error')}")
                return None
            
            # Wait before next check
            time.sleep(5)
        
        print("⏰ Timeout waiting for processing to complete")
        return None

def main():
    """Example usage"""
    # Initialize client
    client = InterviewEvaluationClient()
    
    # Example audio file path (change this to your audio file)
    audio_file = "Audios/ATSID00897933_introduction.wav"
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("Please update the audio_file path in this script.")
        return
    
    # Run evaluation
    results = client.evaluate_audio(audio_file)
    
    if results:
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        # Pretty print the results
        result = results['result']
        print(f"Job ID: {results['job_id']}")
        print(f"Status: {results['status']}")
        print(f"Completed: {results.get('completed_at', 'N/A')}")
        print("\nEvaluation Details:")
        print(f"- Confidence: {result.get('confidence', 'N/A')}")
        print(f"- Pronunciation: {result.get('pronunciation', 'N/A')}")
        print(f"- Fluency: {result.get('fluency', 'N/A')}")
        print(f"- Emotional Tone: {result.get('emotional_tone', 'N/A')}")
        print(f"- Grammar: {result.get('grammar', 'N/A')}")
        print(f"- Suitability Score: {result.get('suitability_score', 'N/A')}/100")
        print(f"- Final Decision: {result.get('final_decision', 'N/A')}")
        
        # Save results to file
        output_file = f"evaluation_results_{results['job_id']}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main() 