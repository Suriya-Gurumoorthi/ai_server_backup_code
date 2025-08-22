#!/usr/bin/env python3
"""
Example script showing how to use LLM models with LangChain and Ollama
"""

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sys

def setup_llm(model_name="llama3.2:3b", temperature=0.7):
    """Setup an LLM with Ollama"""
    llm = Ollama(
        model=model_name,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=temperature
    )
    return llm

def simple_chat(llm, message):
    """Simple chat with the model"""
    print(f"\n🤖 {llm.model}: {message}")
    response = llm(message)
    return response

def create_chain(llm, template):
    """Create a LangChain chain with a custom prompt template"""
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def coding_example():
    """Example for coding tasks"""
    print("\n=== Coding Example ===")
    
    # Use CodeLlama for better coding performance
    llm = setup_llm("codellama:7b", temperature=0.3)
    
    template = """
    You are a helpful coding assistant. Write clean, efficient code for the following request:
    
    {question}
    
    Provide only the code without explanations:
    """
    
    chain = create_chain(llm, template)
    
    question = "Write a Python function to calculate the fibonacci sequence"
    response = chain.run(question)
    return response

def creative_writing_example():
    """Example for creative writing"""
    print("\n=== Creative Writing Example ===")
    
    # Use a larger model for better creative output
    llm = setup_llm("llama3.2:8b", temperature=0.9)
    
    template = """
    You are a creative writer. Write a short story based on this prompt:
    
    {question}
    
    Make it engaging and creative:
    """
    
    chain = create_chain(llm, template)
    
    question = "A robot discovers emotions for the first time"
    response = chain.run(question)
    return response

def conversation_example():
    """Example for conversation"""
    print("\n=== Conversation Example ===")
    
    llm = setup_llm("llama3.2:8b", temperature=0.7)
    
    conversation = [
        "Hello! How are you today?",
        "What's your favorite hobby?",
        "Can you tell me a joke?",
        "What do you think about artificial intelligence?"
    ]
    
    for message in conversation:
        response = simple_chat(llm, message)
        print(f"\n---")

def main():
    print("=== LLM Model Examples ===\n")
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == "chat":
            model = sys.argv[2] if len(sys.argv) > 2 else "llama3.2:3b"
            llm = setup_llm(model)
            while True:
                message = input("\nYou: ")
                if message.lower() in ['quit', 'exit', 'bye']:
                    break
                simple_chat(llm, message)
        
        elif example == "coding":
            coding_example()
        
        elif example == "creative":
            creative_writing_example()
        
        elif example == "conversation":
            conversation_example()
        
        else:
            print("Available examples:")
            print("  python llm_example.py chat [model_name]")
            print("  python llm_example.py coding")
            print("  python llm_example.py creative")
            print("  python llm_example.py conversation")
    
    else:
        print("Choose an example to run:")
        print("1. Interactive chat")
        print("2. Coding example")
        print("3. Creative writing")
        print("4. Conversation example")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            model = input("Enter model name (default: llama3.2:3b): ").strip()
            if not model:
                model = "llama3.2:3b"
            llm = setup_llm(model)
            while True:
                message = input("\nYou: ")
                if message.lower() in ['quit', 'exit', 'bye']:
                    break
                simple_chat(llm, message)
        
        elif choice == "2":
            coding_example()
        
        elif choice == "3":
            creative_writing_example()
        
        elif choice == "4":
            conversation_example()
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 