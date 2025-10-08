# 🏠 Local Ultravox Model Deployment Guide

This guide explains how to deploy your locally downloaded Ultravox model to be accessible from remote VICIdial servers.

## 📋 Overview

Your setup uses a **locally downloaded Ultravox model** from Hugging Face, with a bridge server that allows remote VICIdial servers to connect and use the model for voice conversations.

```
[VICIdial Server] ←→ [Local Ultravox Bridge] ←→ [Local Ultravox Model]
     (Remote)              (Port 9092)              (Hugging Face)
```

## 🚀 Step 1: Setup Local Ultravox Model

### 1.1 Install Dependencies

```bash
# Install required packages for local Ultravox
pip install -r requirements_local.txt

# Or install manually
pip install torch torchaudio transformers ultravox numpy scipy asyncio websockets
```

### 1.2 Run Setup Script

```bash
# Run the automated setup script
python setup_local_ultravox.py
```

This script will:
- Check system requirements (CUDA, memory, etc.)
- Verify Ultravox installation
- Download the model if needed
- Test model loading
- Create configuration files

### 1.3 Manual Model Setup (Alternative)

If you prefer manual setup:

```bash
# Set environment variables
export ULTRAVOX_MODEL_PATH="fixie-ai/ultravox"  # or your local path
export ULTRAVOX_DEVICE="cuda"  # or "cpu"

# Test model loading
python -c "
from ultravox import Ultravox
import torch
model = Ultravox.from_pretrained('$ULTRAVOX_MODEL_PATH', device_map='$ULTRAVOX_DEVICE')
print('Model loaded successfully')
"
```

## 🔧 Step 2: Configure Bridge Server

### 2.1 Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Security settings
export BRIDGE_SECRET_KEY="your-super-secret-key-change-this"
export ALLOWED_IPS="192.168.1.100,192.168.1.101"  # VICIdial server IPs
export ENABLE_AUTH="true"

# Ultravox model settings
export ULTRAVOX_MODEL_PATH="fixie-ai/ultravox"  # or your local path
export ULTRAVOX_DEVICE="cuda"  # or "cpu"

# Audio settings
export VICIDIAL_SAMPLE_RATE=8000
export ULTRAVOX_SAMPLE_RATE=48000
```

### 2.2 Start Local Bridge Server

```bash
# Start the local Ultravox bridge
python openai_local.py
```

You should see output like:
```
INFO - Initializing Ultravox bridge...
INFO - Loading Ultravox model from fixie-ai/ultravox on cuda
INFO - Ultravox model loaded successfully
INFO - Ultravox bridge initialized successfully
INFO - Secure Local Ultravox bridge listening on 0.0.0.0:9092
INFO - Using Ultravox model: fixie-ai/ultravox
INFO - Device: cuda
INFO - Authentication enabled: true
```

### 2.3 Configure Firewall

```bash
# Allow incoming connections on port 9092
sudo ufw allow 9092/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 9092 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

## 🖥️ Step 3: Configure Remote VICIdial Server

### 3.1 Install Client Library

On your VICIdial server, copy the client files:

```bash
# Create directory
mkdir -p /opt/ultravox_client

# Copy client files
# - vicidial_client.py
# - test_remote_setup.py
```

### 3.2 Install Dependencies

```bash
# Install required packages
pip install asyncio numpy websockets aiohttp
```

### 3.3 Test Connection

```bash
# Test connection to your local Ultravox bridge
python test_remote_setup.py YOUR_BRIDGE_SERVER_IP 9092 your-secret-key
```

## 🔒 Step 4: Security Configuration

### 4.1 Network Security

```bash
# Restrict access to specific VICIdial server IPs
export ALLOWED_IPS="192.168.1.100,192.168.1.101"

# Consider using VPN for additional security
```

### 4.2 Authentication

```bash
# Generate a strong secret key
openssl rand -hex 32

# Set on both servers
export BRIDGE_SECRET_KEY="generated-secret-key"
```

### 4.3 SSL/TLS (Optional)

For additional security, you can wrap the connection in SSL:

```python
# In openai_local.py, modify the server creation:
import ssl

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="server.crt", keyfile="server.key")

server = await asyncio.start_server(
    self.handle_vicidial_connection,
    LISTEN_HOST,
    LISTEN_PORT,
    ssl=context
)
```

## 📊 Step 5: Monitoring and Performance

### 5.1 Monitor Model Performance

```bash
# Check GPU usage (if using CUDA)
nvidia-smi

# Monitor system resources
htop
iostat -x 1

# Check bridge server logs
tail -f ultravox_bridge.log
```

### 5.2 Performance Optimization

```bash
# Optimize CUDA settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Optimize model loading
export TORCH_CUDNN_V8_API_ENABLED=1
```

### 5.3 Memory Management

For large models, consider:

```python
# In openai_local.py, add memory optimization
import gc

# After processing audio
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 🧪 Step 6: Testing

### 6.1 Test Local Model

```bash
# Test model loading and basic functionality
python -c "
import torch
from ultravox import Ultravox

model = Ultravox.from_pretrained('fixie-ai/ultravox', device_map='cuda')
print('Model loaded successfully')

# Test with dummy audio
import numpy as np
audio = np.random.randn(48000).astype(np.float32)
print('Model ready for testing')
"
```

### 6.2 Test Bridge Server

```bash
# Start bridge server
python openai_local.py

# In another terminal, test connection
telnet localhost 9092
```

### 6.3 Test Remote Connection

```bash
# On VICIdial server
python test_remote_setup.py YOUR_BRIDGE_SERVER_IP
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Check model path
   ls -la /path/to/ultravox/model
   ```

2. **Out of Memory**
   ```bash
   # Reduce model precision
   export ULTRAVOX_DEVICE="cpu"  # Use CPU instead of GPU
   
   # Or use model quantization
   # Add to openai_local.py:
   model = Ultravox.from_pretrained(
       model_path,
       device_map=device,
       torch_dtype=torch.float16,  # Use half precision
       load_in_8bit=True  # Use 8-bit quantization
   )
   ```

3. **High Latency**
   ```bash
   # Check network latency
   ping YOUR_BRIDGE_SERVER_IP
   
   # Optimize audio buffer sizes
   # Adjust BYTES_PER_FRAME in openai_local.py
   ```

4. **Authentication Failed**
   ```bash
   # Check secret key matches
   echo $BRIDGE_SECRET_KEY
   
   # Check system time synchronization
   ntpdate -s time.nist.gov
   ```

### Debug Commands

```bash
# Check bridge server status
ps aux | grep openai_local

# Monitor network connections
ss -tuln | grep 9092

# Check system resources
free -h
df -h

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## 🔄 Production Deployment

### 1. Systemd Service

Create a systemd service for the local bridge:

```ini
# /etc/systemd/system/ultravox-local-bridge.service
[Unit]
Description=Local Ultravox Bridge Server
After=network.target

[Service]
Type=simple
User=ultravox
Environment=BRIDGE_SECRET_KEY=your-secret-key
Environment=ALLOWED_IPS=192.168.1.100,192.168.1.101
Environment=ENABLE_AUTH=true
Environment=ULTRAVOX_MODEL_PATH=fixie-ai/ultravox
Environment=ULTRAVOX_DEVICE=cuda
WorkingDirectory=/opt/ultravox_bridge
ExecStart=/usr/bin/python3 openai_local.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable ultravox-local-bridge
sudo systemctl start ultravox-local-bridge
sudo systemctl status ultravox-local-bridge
```

### 3. Log Rotation

```bash
# /etc/logrotate.d/ultravox-local-bridge
/var/log/ultravox_local_bridge.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ultravox ultravox
}
```

## 📈 Performance Tips

### 1. Model Optimization

```python
# Use model quantization for better performance
model = Ultravox.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.float16,
    load_in_8bit=True,  # 8-bit quantization
    load_in_4bit=True   # 4-bit quantization (if supported)
)
```

### 2. Audio Processing Optimization

```python
# Use larger audio buffers for better efficiency
AUDIO_BUFFER_SIZE = 2048  # Increase buffer size

# Use batch processing if possible
def process_audio_batch(audio_batch):
    # Process multiple audio frames at once
    pass
```

### 3. Network Optimization

```bash
# Optimize network settings
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
sysctl -p
```

This setup allows you to use your locally downloaded Ultravox model from any remote VICIdial server while maintaining security and performance.




