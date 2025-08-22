# LLM Model Setup Guide

This guide covers different approaches to download and set up Large Language Models (LLMs) for local use.

## 🚀 Quick Start

### 1. Using Ollama (Recommended)

Ollama is the easiest way to run LLMs locally. You already have it installed!

```bash
# Download and run a model
ollama run llama3.2:3b

# Download without running
ollama pull llama3.2:8b

# List installed models
ollama list

# Remove a model
ollama rm model_name
```

### 2. Using the Model Manager Script

I've created a Python script to help you manage models:

```bash
# Interactive mode
python model_manager.py

# Command line usage
python model_manager.py list
python model_manager.py download llama3.2:8b
python model_manager.py test llama3.2:3b "Hello, how are you?"
python model_manager.py recommend
```

### 3. Using LangChain with Ollama

```python
from langchain_community.llms import Ollama

# Setup LLM
llm = Ollama(model="llama3.2:3b", temperature=0.7)

# Use the model
response = llm("Explain quantum computing")
print(response)
```

## 📋 Model Recommendations

### Fast Models (3B parameters)
- `llama3.2:3b` - Good for basic tasks, very fast
- `phi3:mini` - Microsoft's efficient model
- `gemma2:2b` - Google's lightweight model

### Balanced Models (7-8B parameters)
- `llama3.2:8b` - Excellent balance of speed and quality
- `mistral:7b` - Very good performance
- `codellama:7b` - Specialized for coding
- `llama3.2:8b-instruct` - Better for conversations

### High-Quality Models (13B+ parameters)
- `llama3.2:70b` - Best quality, slower
- `codellama:13b` - Advanced coding capabilities
- `llama3.2:8b-instruct` - Good for complex tasks

## 🎯 Use Case Recommendations

### For Coding
```bash
ollama pull codellama:7b
ollama run codellama:7b "Write a Python function to sort a list"
```

### For Conversations
```bash
ollama pull llama3.2:8b-instruct
ollama run llama3.2:8b-instruct "Tell me a story"
```

### For Fast Responses
```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b "What is 2+2?"
```

### For Creative Writing
```bash
ollama pull llama3.2:8b
ollama run llama3.2:8b "Write a poem about technology"
```

## 🔧 Advanced Setup

### Custom Model Configuration

Create a custom model with specific parameters:

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM llama3.2:8b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
SYSTEM "You are a helpful AI assistant."
EOF

# Create the custom model
ollama create my-assistant -f Modelfile
ollama run my-assistant
```

### Using with Python Scripts

```python
# Simple usage
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2:8b")
response = llm("Your prompt here")

# With streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackManager

llm = Ollama(
    model="llama3.2:8b",
    callback_manager=CallbackManager([StreamingStdOutCallbackManager()])
)
response = llm("Your prompt here")
```

### Using with LangChain Chains

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """
You are a helpful assistant. Answer the following question:

{question}

Answer:
"""

prompt = PromptTemplate(input_variables=["question"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("What is machine learning?")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   # Check available models
   ollama list
   
   # Pull the model first
   ollama pull model_name
   ```

2. **Out of memory**
   ```bash
   # Use a smaller model
   ollama run llama3.2:3b
   
   # Or check system resources
   free -h
   ```

3. **Slow responses**
   ```bash
   # Use a faster model
   ollama run phi3:mini
   
   # Or adjust parameters
   ollama run llama3.2:8b --temperature 0.1
   ```

### Performance Tips

1. **Use appropriate model size** for your hardware
2. **Adjust temperature** (0.1 for factual, 0.9 for creative)
3. **Use GPU acceleration** if available
4. **Close other applications** to free up memory

## 📊 Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.2:3b | 3B | ⚡⚡⚡ | ⭐⭐ | Fast responses, basic tasks |
| llama3.2:8b | 8B | ⚡⚡ | ⭐⭐⭐⭐ | General purpose, good balance |
| llama3.2:70b | 70B | ⚡ | ⭐⭐⭐⭐⭐ | High quality, complex tasks |
| codellama:7b | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Coding tasks |
| mistral:7b | 7B | ⚡⚡ | ⭐⭐⭐⭐ | General purpose |
| phi3:mini | 3.8B | ⚡⚡⚡ | ⭐⭐⭐ | Fast, efficient |

## 🎮 Interactive Examples

Run the example script to try different use cases:

```bash
# Interactive chat
python llm_example.py chat llama3.2:8b

# Coding example
python llm_example.py coding

# Creative writing
python llm_example.py creative

# Conversation example
python llm_example.py conversation
```

## 🔗 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Models](https://huggingface.co/models)
- [Model Comparison](https://ollama.ai/library)

## 💡 Tips for Best Results

1. **Start with smaller models** to test your setup
2. **Experiment with different temperatures** for different tasks
3. **Use specific models** for specific tasks (e.g., CodeLlama for coding)
4. **Monitor system resources** when running larger models
5. **Keep models updated** with `ollama pull model_name`

Happy modeling! 🚀 