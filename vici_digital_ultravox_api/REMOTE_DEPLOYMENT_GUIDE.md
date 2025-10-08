# 🌐 Remote Ultravox Bridge Deployment Guide

This guide explains how to deploy your locally installed Ultravox model to be accessible from remote VICIdial servers.

## 📋 Overview

Your current setup allows remote VICIdial servers to connect to your Ultravox model server via a secure bridge. The architecture is:

```
[VICIdial Server] ←→ [Ultravox Bridge Server] ←→ [Ultravox Model]
     (Remote)              (Your Server)           (Local)
```

## 🚀 Step 1: Configure Your Ultravox Bridge Server

### 1.1 Start the Secure Bridge

```bash
# Set environment variables for security
export BRIDGE_SECRET_KEY="your-super-secret-key-change-this"
export ALLOWED_IPS="192.168.1.100,192.168.1.101"  # VICIdial server IPs
export ENABLE_AUTH="true"

# Start the secure bridge
python openai_secure.py
```

### 1.2 Configure Firewall

```bash
# Allow incoming connections on port 9092
sudo ufw allow 9092/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 9092 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

### 1.3 Verify Bridge is Running

```bash
# Check if bridge is listening
netstat -tlnp | grep 9092

# Test local connection
telnet localhost 9092
```

## 🖥️ Step 2: Configure Remote VICIdial Server

### 2.1 Install Client Library

Copy the client files to your VICIdial server:

```bash
# On VICIdial server
mkdir -p /opt/ultravox_client
# Copy vicidial_client.py to this directory
```

### 2.2 Install Dependencies

```bash
# Install required packages
pip install asyncio numpy websockets aiohttp
```

### 2.3 Test Connection

```bash
# Test connection to your bridge server
python vicidial_client.py YOUR_BRIDGE_SERVER_IP
```

## 🔧 Step 3: VICIdial Integration

### 3.1 Basic Integration Example

```python
# vicidial_ultravox_integration.py
import asyncio
from vicidial_client import VicidialUltravoxIntegration

class VicidialUltravoxHandler:
    def __init__(self):
        self.integration = VicidialUltravoxIntegration(
            bridge_host="YOUR_BRIDGE_SERVER_IP",
            bridge_port=9092,
            secret_key="your-super-secret-key-change-this"
        )
    
    async def start_call(self):
        """Start a new call with Ultravox"""
        await self.integration.start()
        
        # Your VICIdial call logic here
        # This is where you'd integrate with VICIdial's audio pipeline
        
    async def stop_call(self):
        """End the call"""
        await self.integration.stop()
    
    async def send_audio_to_ultravox(self, audio_data):
        """Send audio from VICIdial to Ultravox"""
        await self.integration.send_vicidial_audio(audio_data)
```

### 3.2 Advanced VICIdial Integration

For production use, you'll need to integrate with VICIdial's specific audio pipeline:

```python
# Example integration with VICIdial's AGI system
import sys
import asyncio
from vicidial_client import VicidialUltravoxIntegration

class VicidialAGIHandler:
    def __init__(self):
        self.integration = None
        self.call_active = False
    
    async def handle_call(self, channel, context, extension, priority):
        """Handle incoming VICIdial call"""
        try:
            # Initialize Ultravox integration
            self.integration = VicidialUltravoxIntegration(
                bridge_host="YOUR_BRIDGE_SERVER_IP",
                bridge_port=9092,
                secret_key="your-super-secret-key-change-this"
            )
            
            # Start integration
            await self.integration.start()
            self.call_active = True
            
            # Set up audio callback for Ultravox responses
            self.integration.bridge_client.set_audio_callback(
                self.handle_ultravox_response
            )
            
            # Answer the call
            channel.answer()
            
            # Main call loop
            while self.call_active:
                # Read audio from VICIdial
                vicidial_audio = await self.read_vicidial_audio(channel)
                
                if vicidial_audio:
                    # Send to Ultravox
                    await self.integration.send_vicidial_audio(vicidial_audio)
                
                # Check for hangup
                if channel.get_state() == 'Down':
                    break
                    
                await asyncio.sleep(0.02)  # 20ms delay
                
        except Exception as e:
            print(f"Error in call handler: {e}")
        finally:
            await self.cleanup_call()
    
    def handle_ultravox_response(self, audio_data):
        """Handle audio response from Ultravox"""
        # Send audio back to VICIdial
        asyncio.create_task(self.send_audio_to_vicidial(audio_data))
    
    async def send_audio_to_vicidial(self, audio_data):
        """Send audio to VICIdial channel"""
        # Implementation depends on your VICIdial setup
        # This might involve writing to a file or using VICIdial's API
        pass
    
    async def read_vicidial_audio(self, channel):
        """Read audio from VICIdial channel"""
        # Implementation depends on your VICIdial setup
        # This might involve reading from a file or using VICIdial's API
        pass
    
    async def cleanup_call(self):
        """Clean up call resources"""
        if self.integration:
            await self.integration.stop()
        self.call_active = False
```

## 🔒 Step 4: Security Configuration

### 4.1 Network Security

```bash
# On bridge server - restrict access to specific IPs
export ALLOWED_IPS="192.168.1.100,192.168.1.101"

# Use VPN for additional security
# Consider setting up a VPN between VICIdial and bridge servers
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
# In openai_secure.py, modify the server creation:
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

## 📊 Step 5: Monitoring and Logging

### 5.1 Enable Detailed Logging

```python
# In your bridge server
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultravox_bridge.log'),
        logging.StreamHandler()
    ]
)
```

### 5.2 Monitor Connection Status

```bash
# Check active connections
netstat -an | grep 9092

# Monitor logs
tail -f ultravox_bridge.log
```

## 🧪 Step 6: Testing

### 6.1 Test Bridge Server

```bash
# On bridge server
python openai_secure.py

# In another terminal
telnet localhost 9092
```

### 6.2 Test Remote Connection

```bash
# On VICIdial server
python vicidial_client.py YOUR_BRIDGE_SERVER_IP
```

### 6.3 Test Full Integration

```python
# Test script for full integration
import asyncio
from vicidial_client import VicidialUltravoxIntegration

async def test_full_integration():
    integration = VicidialUltravoxIntegration(
        bridge_host="YOUR_BRIDGE_SERVER_IP",
        bridge_port=9092,
        secret_key="your-secret-key"
    )
    
    try:
        await integration.start()
        print("Integration started successfully")
        
        # Send test audio
        test_audio = b'\x00' * 320  # 20ms of silence
        await integration.send_vicidial_audio(test_audio)
        
        # Wait for response
        await asyncio.sleep(5)
        
    finally:
        await integration.stop()
        print("Integration stopped")

if __name__ == "__main__":
    asyncio.run(test_full_integration())
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check firewall settings
   - Verify bridge server is running
   - Check IP address and port

2. **Authentication Failed**
   - Verify secret key matches on both servers
   - Check system time synchronization
   - Ensure ENABLE_AUTH is set correctly

3. **Audio Quality Issues**
   - Check network latency
   - Verify sample rate conversion
   - Monitor CPU usage on bridge server

4. **High Latency**
   - Use dedicated network connection
   - Consider geographical proximity
   - Optimize audio buffer sizes

### Debug Commands

```bash
# Check bridge server status
ps aux | grep openai_secure

# Monitor network connections
ss -tuln | grep 9092

# Check system resources
htop
iostat -x 1

# Test network connectivity
ping YOUR_BRIDGE_SERVER_IP
telnet YOUR_BRIDGE_SERVER_IP 9092
```

## 📈 Performance Optimization

### 1. Network Optimization

```bash
# Optimize network settings
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
sysctl -p
```

### 2. Audio Processing Optimization

```python
# Use numpy for faster audio processing
import numpy as np

# Optimize buffer sizes
AUDIO_BUFFER_SIZE = 1024  # Adjust based on your needs
```

### 3. Resource Monitoring

```bash
# Monitor system resources
watch -n 1 'echo "CPU:" && top -bn1 | grep "Cpu(s)" && echo "Memory:" && free -h && echo "Network:" && netstat -i'
```

## 🔄 Production Deployment

### 1. Systemd Service

Create a systemd service for the bridge:

```ini
# /etc/systemd/system/ultravox-bridge.service
[Unit]
Description=Ultravox Bridge Server
After=network.target

[Service]
Type=simple
User=ultravox
Environment=BRIDGE_SECRET_KEY=your-secret-key
Environment=ALLOWED_IPS=192.168.1.100,192.168.1.101
Environment=ENABLE_AUTH=true
WorkingDirectory=/opt/ultravox_bridge
ExecStart=/usr/bin/python3 openai_secure.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable ultravox-bridge
sudo systemctl start ultravox-bridge
sudo systemctl status ultravox-bridge
```

### 3. Log Rotation

```bash
# /etc/logrotate.d/ultravox-bridge
/var/log/ultravox_bridge.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ultravox ultravox
}
```

This setup allows you to use your locally installed Ultravox model from any remote VICIdial server while maintaining security and performance.




