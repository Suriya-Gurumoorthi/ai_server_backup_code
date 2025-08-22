# 🚀 Project Overview: AI-Powered Voice & Interview System

## 📋 Project Summary

This is a comprehensive AI project that combines multiple cutting-edge technologies:

1. **Voice-to-Voice AI System** - Real-time conversations using Piper TTS and Ultravox STT
2. **AI Interview Evaluation** - Professional candidate assessment using audio analysis
3. **Hybrid Search & RAG** - Document processing with advanced retrieval systems
4. **Audio Processing Pipeline** - MP3 to WAV conversion and management
5. **LLM Integration** - Local AI models via Ollama

## 🏗️ Architecture Overview

```
Project Root/
├── 🎤 voice_to_voice/          # Real-time voice AI system
├── 🎯 AI_Interview/            # Interview evaluation system
├── 🔍 hybrid_search/           # Document search & RAG
├── 🎵 Audio Processing/        # Audio conversion & management
├── 🤖 LLM Integration/         # Ollama & LangChain setup
└── 🛠️ Utilities & Scripts/     # Helper tools & examples
```

## 🎤 Voice-to-Voice System

**Location**: `voice_to_voice/`

**Purpose**: Enables real-time AI conversations with natural voice input/output

**Key Features**:
- Real-time speech recognition (Ultravox STT)
- Natural voice synthesis (Piper TTS)
- Interruption handling
- Web interface for testing
- Role-based AI personalities

**Usage**:
```bash
cd voice_to_voice
python main.py
# Access web interface at http://localhost:5000
```

## 🎯 AI Interview Evaluation

**Location**: `AI_Interview/`

**Purpose**: Professional evaluation of candidate communication skills

**Key Features**:
- Audio file processing (WAV, MP3, FLAC, M4A)
- Comprehensive evaluation criteria
- Web API with FastAPI
- Beautiful drag-and-drop interface
- Background processing

**Usage**:
```bash
cd AI_Interview
python start_api.py
# Access web interface at http://localhost:8000
```

## 🔍 Hybrid Search & RAG

**Location**: `hybrid_search/`

**Purpose**: Advanced document search with multiple retrieval methods

**Key Features**:
- Multi-Query Retrieval (MQR)
- Maximum Marginal Relevance (MMR)
- ChromaDB vector storage
- LangChain integration
- PDF document processing

**Usage**:
```bash
cd hybrid_search
python hybrid_search.py
```

## 🎵 Audio Processing

**Location**: Root directory

**Purpose**: Audio file management and conversion

**Key Features**:
- MP3 to WAV conversion
- Batch processing
- Progress tracking
- Error handling
- FFmpeg integration

**Usage**:
```bash
python audio_downloader.py
# Or use the example script
python example_usage.py
```

## 🤖 LLM Integration (Ollama)

**Location**: Root directory

**Purpose**: Local AI model management and integration

**Key Features**:
- Ollama model management
- LangChain integration
- Multiple model support
- Interactive testing
- API access

### 🚀 Quick Start with Ollama

#### 1. Check Ollama Status
```bash
# Check if Ollama is running
ollama ps

# List installed models
ollama list

# Check Ollama service
ps aux | grep ollama
```

#### 2. Use Ollama Directly
```bash
# Run a model interactively
ollama run llama3.2:3b

# Test with a prompt
ollama run llama3.2:3b "What is artificial intelligence?"

# Use the API
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3.2:3b", "prompt": "Hello!"}'
```

#### 3. Use Python Integration
```bash
# Activate virtual environment
source venv/bin/activate

# Test model manager
python model_manager.py list
python model_manager.py test llama3.2:3b "Hello!"

# Run examples
python llm_example.py chat llama3.2:3b
python llm_example.py coding
python llm_example.py creative

# Run comprehensive test
python test_ollama_integration.py
```

### 📚 Available Models

**Currently Installed**:
- `llama3.2:3b` (2.0 GB) - Fast, lightweight model

**Recommended Models**:
```bash
# Fast models (3B parameters)
ollama pull llama3.2:3b      # Already installed
ollama pull phi3:mini        # Microsoft's efficient model
ollama pull gemma2:2b        # Google's lightweight model

# Balanced models (7-8B parameters)
ollama pull llama3.2:8b      # Excellent balance
ollama pull mistral:7b       # Very good performance
ollama pull codellama:7b     # Specialized for coding

# High-quality models (13B+ parameters)
ollama pull llama3.2:70b     # Best quality, slower
ollama pull codellama:13b    # Advanced coding capabilities
```

### 🔧 Ollama Configuration

**Service Management**:
```bash
# Start Ollama service
sudo systemctl start ollama

# Enable auto-start
sudo systemctl enable ollama

# Check status
sudo systemctl status ollama
```

**Model Management**:
```bash
# Download a model
ollama pull model_name

# Remove a model
ollama rm model_name

# List all models
ollama list

# Run a model
ollama run model_name
```

## 🛠️ Development Setup

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ollama Installation
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve

# Download first model
ollama pull llama3.2:3b
```

### 3. Testing Setup
```bash
# Test Ollama
ollama run llama3.2:3b "Hello!"

# Test Python integration
python test_ollama_integration.py

# Test model manager
python model_manager.py interactive
```

## 📊 System Requirements

**Minimum**:
- 8GB RAM
- 4GB free disk space
- Python 3.8+

**Recommended**:
- 16GB+ RAM
- 10GB+ free disk space
- GPU acceleration (optional)
- Python 3.10+

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   sudo systemctl start ollama
   # or
   ollama serve
   ```

2. **Model not found**:
   ```bash
   ollama pull model_name
   ```

3. **Out of memory**:
   - Use smaller models (3B instead of 70B)
   - Close other applications
   - Check system resources: `free -h`

4. **LangChain import errors**:
   ```bash
   pip install --upgrade langchain langchain-community
   ```

### Performance Tips

1. **Model Selection**:
   - Use 3B models for fast responses
   - Use 8B models for balanced performance
   - Use 70B models for high quality (if you have resources)

2. **System Optimization**:
   - Close unnecessary applications
   - Use SSD storage for models
   - Enable GPU acceleration if available

## 🎯 Use Cases

### 1. **Real Estate Agent AI**
- Role: Customer service and property consultation
- Use: `voice_to_voice/` system
- Model: `llama3.2:8b` for balanced performance

### 2. **Interview Evaluation**
- Role: HR candidate assessment
- Use: `AI_Interview/` system
- Model: `llama3.2:3b` for fast processing

### 3. **Document Analysis**
- Role: Legal/contract document processing
- Use: `hybrid_search/` system
- Model: `codellama:7b` for structured output

### 4. **Creative Writing**
- Role: Content generation and brainstorming
- Use: Direct Ollama integration
- Model: `llama3.2:8b` with high temperature

## 🚀 Next Steps

1. **Explore Voice System**: Test the voice-to-voice AI
2. **Try Interview System**: Upload audio files for evaluation
3. **Test Document Search**: Process PDFs with hybrid search
4. **Experiment with Models**: Try different Ollama models
5. **Customize Prompts**: Adapt AI behavior for your needs

## 📞 Support & Resources

- **Ollama Documentation**: https://ollama.ai/docs
- **LangChain Documentation**: https://python.langchain.com/
- **Project Issues**: Check individual component READMEs
- **Model Recommendations**: Use `python model_manager.py recommend`

---

**🎉 You're all set! Ollama is running and integrated with your project. Start exploring the different AI capabilities available to you!**
