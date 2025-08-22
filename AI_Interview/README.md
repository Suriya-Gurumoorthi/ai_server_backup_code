# AI Interview Evaluation System

A modular system for evaluating interview audio using the Ultravox model. This system provides professional AI-powered evaluation of candidate communication skills during interviews, with both command-line and web API interfaces.

## 🏗️ Project Structure

```
AI_Interview/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── __init__.py
│   │   └── ultravox_model.py
│   ├── processors/        # Audio processing logic
│   │   ├── __init__.py
│   │   └── audio_processor.py
│   └── utils/             # Utility functions
│       └── __init__.py
├── api/                   # Web API
│   ├── templates/         # HTML templates
│   │   └── index.html     # Web interface
│   └── app.py             # FastAPI application
├── scripts/               # Executable scripts
│   ├── load_model.py      # Pre-load model into VRAM
│   └── quick_process.py   # Quick audio processing
├── configs/               # Configuration files
│   └── config.py          # Centralized settings
├── Audios/                # Audio files directory
├── uploads/               # Temporary upload directory
├── outputs/               # Generated outputs
├── logs/                  # Log files
├── main.py                # Main CLI application
├── start_api.py           # API startup script
├── api_client_example.py  # Example API client
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web API
```bash
python start_api.py
```

### 3. Access the Web Interface
Open your browser and go to: **http://localhost:8000**

## 🌐 Web API Usage

### Web Interface
- **URL**: http://localhost:8000
- **Features**: Drag-and-drop file upload, real-time processing status, beautiful results display
- **Supported Formats**: WAV, MP3, FLAC, M4A (Max 50MB)

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns API health and model status.

#### Upload Audio
```bash
POST /upload
Content-Type: multipart/form-data
```
Upload an audio file for evaluation. Returns a job ID for tracking.

#### Check Status
```bash
GET /status/{job_id}
```
Get processing status for a specific job.

#### Get Results
```bash
GET /results/{job_id}
```
Get evaluation results for a completed job.

#### Delete Job
```bash
DELETE /jobs/{job_id}
```
Delete a job and its status from memory.

### API Documentation
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

## 💻 Command Line Usage

### Method 1: Main Application
```bash
# Process a single audio file
python main.py Audios/interview.wav

# Save results to file
python main.py Audios/interview.wav --output results.txt

# Verbose output
python main.py Audios/interview.wav --verbose
```

### Method 2: Quick Script
```bash
# Edit audio_path in scripts/quick_process.py
# Then run:
python scripts/quick_process.py
```

### Method 3: API Client
```bash
# Run the example client
python api_client_example.py
```

## 🔧 API Client Example

```python
from api_client_example import InterviewEvaluationClient

# Initialize client
client = InterviewEvaluationClient("http://localhost:8000")

# Evaluate audio file
results = client.evaluate_audio("path/to/audio.wav")

if results:
    print(f"Score: {results['result']['suitability_score']}/100")
    print(f"Decision: {results['result']['final_decision']}")
```

## 📁 Directory Descriptions

### `src/models/`
Contains the Ultravox model implementation with singleton pattern for efficient memory usage.

### `src/processors/`
Audio processing logic and evaluation prompts.

### `src/utils/`
Utility functions and helpers (expandable).

### `api/`
- `app.py`: FastAPI application with all endpoints
- `templates/index.html`: Modern web interface with drag-and-drop

### `scripts/`
- `load_model.py`: Pre-loads the model into VRAM
- `quick_process.py`: Simple script for processing single audio files

### `configs/`
Centralized configuration for model parameters, audio settings, and evaluation criteria.

## ⚙️ Configuration

Edit `configs/config.py` to customize:
- Model parameters
- Audio processing settings
- Evaluation criteria weights
- File paths

## 🔧 Features

- ✅ **Web API**: RESTful API with FastAPI
- ✅ **Beautiful UI**: Modern, responsive web interface
- ✅ **Real-time Status**: Live processing status updates
- ✅ **Drag & Drop**: Easy file upload interface
- ✅ **Background Processing**: Non-blocking audio processing
- ✅ **Efficient Model Loading**: Singleton pattern prevents reloading
- ✅ **Modular Design**: Clean separation of concerns
- ✅ **Multiple Audio Formats**: Supports WAV, MP3, FLAC, M4A
- ✅ **Professional Evaluation**: Comprehensive interview assessment
- ✅ **Error Handling**: Robust error handling and validation
- ✅ **Configurable**: Easy to customize settings

## 📊 Evaluation Criteria

The system evaluates candidates on:
1. **Confidence** (tone, hesitation, filler words)
2. **Pronunciation & Clarity**
3. **Fluency & Logical Coherence**
4. **Emotional Tone** (nervous, calm, assertive)
5. **Grammar & Vocabulary Usage**
6. **Overall Suitability Score** (0-100)
7. **Final Decision** (Hire/Reject)

## 🛠️ Development

### Adding New Features
1. Add new modules to appropriate `src/` subdirectories
2. Update `__init__.py` files for proper imports
3. Add configuration options to `configs/config.py`
4. Update API endpoints in `api/app.py` if needed
5. Update this README

### Testing
```bash
# Test model loading
python scripts/load_model.py

# Test CLI processing
python main.py Audios/test_audio.wav

# Test API
python start_api.py
# Then visit http://localhost:8000
```

### API Development
```bash
# Start API with auto-reload
python start_api.py

# Test API endpoints
curl http://localhost:8000/health
```

## 🔒 Security Notes

- The API currently allows CORS from all origins for development
- In production, configure CORS properly
- File uploads are validated for type and size
- Temporary files are automatically cleaned up

## 📝 License

This project is for internal use only.

## 🤝 Contributing

1. Follow the modular structure
2. Add proper documentation
3. Update configuration as needed
4. Test thoroughly before committing

---

**Note**: The Ultravox model is loaded only once per session for optimal performance. Subsequent audio processing reuses the loaded model from VRAM. 