#!/usr/bin/env python3
"""
LLM Model Manager - A simple tool to download, list, and test Ollama models
"""

import subprocess
import json
import sys
from typing import List, Dict

class OllamaModelManager:
    def __init__(self):
        self.available_models = {
            "fast": ["llama3.2:3b", "phi3:mini"],
            "balanced": ["llama3.2:8b", "mistral:7b", "codellama:7b"],
            "high_quality": ["llama3.2:70b", "llama3.2:8b-instruct"],
            "coding": ["codellama:7b", "codellama:13b", "llama3.2:8b"]
        }
    
    def run_command(self, command: List[str]) -> str:
        """Run a shell command and return the output"""
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running command {' '.join(command)}: {e}")
            return ""
    
    def list_installed_models(self) -> List[str]:
        """List all installed models"""
        output = self.run_command(["ollama", "list"])
        models = []
        for line in output.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    
    def download_model(self, model_name: str) -> bool:
        """Download a specific model"""
        print(f"Downloading {model_name}...")
        output = self.run_command(["ollama", "pull", model_name])
        if output:
            print(f"Successfully downloaded {model_name}")
            return True
        else:
            print(f"Failed to download {model_name}")
            return False
    
    def test_model(self, model_name: str, prompt: str = "Hello, how are you?") -> str:
        """Test a model with a simple prompt"""
        print(f"Testing {model_name} with prompt: '{prompt}'")
        try:
            result = subprocess.run(
                ["ollama", "run", model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Timeout - model took too long to respond"
        except Exception as e:
            return f"Error: {e}"
    
    def show_recommendations(self):
        """Show model recommendations for different use cases"""
        print("\n=== Model Recommendations ===")
        for category, models in self.available_models.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for model in models:
                print(f"  - {model}")
    
    def interactive_setup(self):
        """Interactive setup for downloading and testing models"""
        print("=== LLM Model Manager ===\n")
        
        # Show current models
        installed = self.list_installed_models()
        print(f"Currently installed models: {', '.join(installed) if installed else 'None'}")
        
        # Show recommendations
        self.show_recommendations()
        
        while True:
            print("\nOptions:")
            print("1. Download a model")
            print("2. Test a model")
            print("3. List installed models")
            print("4. Show recommendations")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                model_name = input("Enter model name to download: ").strip()
                if model_name:
                    self.download_model(model_name)
            
            elif choice == "2":
                model_name = input("Enter model name to test: ").strip()
                if model_name:
                    prompt = input("Enter test prompt (or press Enter for default): ").strip()
                    if not prompt:
                        prompt = "Hello, how are you?"
                    response = self.test_model(model_name, prompt)
                    print(f"\nResponse: {response}")
            
            elif choice == "3":
                installed = self.list_installed_models()
                print(f"Installed models: {', '.join(installed) if installed else 'None'}")
            
            elif choice == "4":
                self.show_recommendations()
            
            elif choice == "5":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    manager = OllamaModelManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            models = manager.list_installed_models()
            print("Installed models:", ', '.join(models) if models else 'None')
        
        elif command == "download" and len(sys.argv) > 2:
            model_name = sys.argv[2]
            manager.download_model(model_name)
        
        elif command == "test" and len(sys.argv) > 2:
            model_name = sys.argv[2]
            prompt = sys.argv[3] if len(sys.argv) > 3 else "Hello, how are you?"
            response = manager.test_model(model_name, prompt)
            print(f"Response: {response}")
        
        elif command == "recommend":
            manager.show_recommendations()
        
        else:
            print("Usage:")
            print("  python model_manager.py list")
            print("  python model_manager.py download <model_name>")
            print("  python model_manager.py test <model_name> [prompt]")
            print("  python model_manager.py recommend")
            print("  python model_manager.py interactive")
    else:
        manager.interactive_setup()

if __name__ == "__main__":
    main() 