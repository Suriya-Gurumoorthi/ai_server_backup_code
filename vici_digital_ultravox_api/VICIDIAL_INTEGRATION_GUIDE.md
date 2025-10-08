# 🎯 Ultravox Vicidial Integration Guide

This guide provides comprehensive instructions for integrating Ultravox with a Vicidial server using API key authentication for voice processing and call management.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [API Endpoints](#api-endpoints)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

## 🔧 Prerequisites

### System Requirements
- Python 3.8 or higher
- Linux/Unix environment (recommended)
- CUDA-compatible GPU (optional, for faster processing)
- At least 8GB RAM
- 10GB free disk space

### Vicidial Server Requirements
- Vicidial server with API access enabled
- Valid API key with appropriate permissions
- Campaigns configured for testing
- Audio streaming capabilities

### Ultravox Requirements
- Ultravox model loaded and accessible
- Required Python packages installed
- Audio processing libraries

## 📦 Installation

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements_vicidial_test.txt

# Install additional Ultravox dependencies
pip install -r requirements_ultravox.txt
```

### 2. Verify Ultravox Installation

```bash
# Check if Ultravox model is available
python check_ultravox_models.py
```

### 3. Set Up Configuration

```bash
# Copy and edit the configuration template
cp vicidial_config.json my_vicidial_config.json
nano my_vicidial_config.json
```

## ⚙️ Configuration

### Configuration File Structure

```json
{
  "vicidial": {
    "server_url": "https://your-vicidial-server.com",
    "api_key": "your_api_key_here",
    "username": "admin",
    "timeout": 30
  },
  "test_config": {
    "test_phone_number": "+1234567890",
    "campaign_id": "your_campaign_id",
    "conversation_duration": 30,
    "reference_audio_file": "path/to/reference_audio.wav",
    "cloning_text": "Hello, this is a test of voice cloning capabilities."
  },
  "ultravox": {
    "model_path": "/path/to/ultravox/model",
    "device": "cuda",
    "max_tokens": 100
  },
  "logging": {
    "level": "INFO",
    "file": "ultravox_vicidial_test.log"
  }
}
```

### Required Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| `vicidial.server_url` | Your Vicidial server URL | `https://vicidial.example.com` |
| `vicidial.api_key` | Your API key for authentication | `abc123def456ghi789` |
| `vicidial.username` | Username for API access | `admin` |
| `test_config.test_phone_number` | Phone number for testing | `+1234567890` |
| `test_config.campaign_id` | Campaign ID for calls | `CAMPAIGN_001` |

## 🚀 Usage

### Quick Start

1. **Basic Connection Test**
```bash
python quick_vicidial_test.py
```

2. **Comprehensive Testing**
```bash
python ultravox_vicidial_test.py --config my_vicidial_config.json --test-type comprehensive
```

3. **Basic Testing Only**
```bash
python ultravox_vicidial_test.py --config my_vicidial_config.json --test-type basic
```

### Command Line Options

```bash
python ultravox_vicidial_test.py [OPTIONS]

Options:
  --config CONFIG_FILE    Path to configuration file (required)
  --test-type TYPE        Type of test: basic, comprehensive, custom
  --output OUTPUT_FILE    Output file for test results
  --help                  Show help message
```

## 🔌 API Endpoints

The integration supports the following Vicidial API endpoints:

### Health Check
```
GET /api/health
```
- **Purpose**: Verify server connectivity
- **Response**: Server status and version information

### Campaigns
```
GET /api/campaigns
```
- **Purpose**: Retrieve available campaigns
- **Response**: List of campaigns with IDs and names

### Call Management
```
POST /api/calls/initiate
GET /api/calls/{call_id}/status
POST /api/calls/{call_id}/end
```
- **Purpose**: Initiate, monitor, and end calls
- **Parameters**: phone_number, campaign_id, api_key

### Audio Streaming
```
GET /api/calls/{call_id}/audio
```
- **Purpose**: Retrieve call audio data
- **Parameters**: start_time, end_time (optional)

## 🧪 Testing

### Test Scenarios

1. **Connection Test**
   - Verifies connectivity to Vicidial server
   - Checks Ultravox model availability
   - Validates API key authentication

2. **Campaign Retrieval**
   - Lists available campaigns
   - Validates campaign access permissions

3. **Call Initiation**
   - Initiates test calls
   - Validates call creation process
   - Checks call status monitoring

4. **Real-time Audio Processing**
   - Streams call audio
   - Processes audio with Ultravox
   - Tests transcription capabilities

5. **Voice Cloning**
   - Tests voice cloning functionality
   - Validates TTS generation
   - Checks audio quality

### Running Tests

```bash
# Run all tests
python ultravox_vicidial_test.py --config config.json --test-type comprehensive

# Run basic tests only
python ultravox_vicidial_test.py --config config.json --test-type basic

# Save results to specific file
python ultravox_vicidial_test.py --config config.json --output test_results.json
```

### Test Results

Test results are saved in JSON format with the following structure:

```json
{
  "test_start_time": "2025-01-27T10:30:00",
  "tests": {
    "basic_connection": true,
    "get_campaigns": true,
    "call_initiation": true,
    "real_time_conversation": true,
    "voice_cloning": true
  },
  "overall_status": "passed",
  "summary": "5/5 tests passed",
  "test_end_time": "2025-01-27T10:35:00"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   ❌ Failed to connect to Vicidial server
   ```
   - **Solution**: Check server URL and network connectivity
   - **Verify**: API key is valid and has proper permissions

2. **Authentication Error**
   ```
   ❌ Vicidial server returned status 401
   ```
   - **Solution**: Verify API key in configuration
   - **Check**: Username and permissions

3. **No Campaigns Available**
   ```
   ⚠️ No campaigns available for testing
   ```
   - **Solution**: Create campaigns in Vicidial
   - **Check**: User has campaign access permissions

4. **Call Initiation Failed**
   ```
   ❌ Failed to initiate call: 400
   ```
   - **Solution**: Verify phone number format
   - **Check**: Campaign is active and accessible

5. **Ultravox Model Not Loaded**
   ```
   ❌ Ultravox model not loaded
   ```
   - **Solution**: Load Ultravox model first
   - **Run**: `python model_loader.py`

### Debug Mode

Enable debug logging by modifying the configuration:

```json
{
  "logging": {
    "level": "DEBUG",
    "file": "debug_vicidial_test.log"
  }
}
```

### Log Files

- **Main log**: `ultravox_vicidial_test.log`
- **Debug log**: `debug_vicidial_test.log` (when debug mode enabled)
- **Test results**: `ultravox_vicidial_test_results_YYYYMMDD_HHMMSS.json`

## 🚀 Advanced Features

### Custom Test Scenarios

Create custom test scenarios by extending the `UltravoxVicidialTester` class:

```python
class CustomTester(UltravoxVicidialTester):
    def test_custom_scenario(self):
        """Custom test scenario"""
        # Your custom test logic here
        pass
```

### Real-time Audio Processing

The integration supports real-time audio processing:

```python
# Process audio stream in real-time
audio_data = vicidial.get_call_audio(call_id)
if audio_data:
    result = ultravox.process_audio(audio_data)
    # Handle processing result
```

### Voice Cloning Integration

Integrate with voice cloning systems:

```python
# Clone voice and generate TTS
cloned_audio = voice_cloner.clone_voice(reference_audio, text)
# Send to Vicidial for playback
```

### Batch Testing

Run multiple tests in sequence:

```bash
# Create batch test script
for phone in $(cat phone_numbers.txt); do
    python ultravox_vicidial_test.py --config config.json --test-phone $phone
done
```

## 📞 Support

### Getting Help

1. **Check logs**: Review log files for detailed error information
2. **Verify configuration**: Ensure all required fields are properly set
3. **Test connectivity**: Use `quick_vicidial_test.py` for basic connectivity
4. **Check permissions**: Verify API key has required permissions

### Common Commands

```bash
# Check Ultravox model status
python check_ultravox_models.py

# Test basic connectivity
python quick_vicidial_test.py

# Run comprehensive tests
python ultravox_vicidial_test.py --config config.json --test-type comprehensive

# View logs
tail -f ultravox_vicidial_test.log
```

## 📝 License

This integration script is provided as-is for testing and development purposes. Ensure compliance with your Vicidial server's terms of service and API usage policies.

---

**🎉 Happy Testing!** 

For additional support or feature requests, please refer to the project documentation or contact your system administrator.
