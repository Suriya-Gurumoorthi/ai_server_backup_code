# Voice-to-Voice AI System - Network Access Guide

## 🚀 **Server Status**
- **Main Application**: Running on `http://10.80.2.40:5000/`
- **Test Page**: Available on `http://10.80.2.40:8081/test_mic.html`

## 📱 **How to Access from Any Device**

### **From Your Computer:**
1. **Main App**: Open browser and go to `http://10.80.2.40:5000/`
2. **Test Page**: Open browser and go to `http://10.80.2.40:8081/test_mic.html`

### **From Your Phone/Tablet:**
1. Make sure your device is connected to the same WiFi network
2. Open browser and go to `http://10.80.2.40:5000/`
3. For testing: `http://10.80.2.40:8081/test_mic.html`

### **From Other Computers on Network:**
1. Open browser and go to `http://10.80.2.40:5000/`

## 🔧 **Troubleshooting Network Access**

### **If Microphone Doesn't Work:**
1. **Try the test page first**: `http://10.80.2.40:8081/test_mic.html`
2. **Check browser permissions**: Allow microphone access when prompted
3. **Use Chrome/Firefox**: These browsers work best with network access

### **If You Can't Connect:**
1. **Check firewall**: Make sure ports 5000 and 8081 are open
2. **Check network**: Ensure devices are on the same network
3. **Try localhost**: If network doesn't work, use `http://127.0.0.1:5000/`

## 🎤 **Testing Steps**

### **Step 1: Test Microphone**
1. Go to `http://10.80.2.40:8081/test_mic.html`
2. Click "Test Microphone"
3. Allow microphone access when prompted
4. Click "Start Recording (3s)" and speak
5. You should see "✅ Recording successful!"

### **Step 2: Use Main Application**
1. Go to `http://10.80.2.40:5000/`
2. Select an AI role from the sidebar
3. Click "Start Conversation"
4. Click "Start Recording"
5. Speak your message
6. Click "Stop Recording"

## 🔒 **Browser Permissions**

### **Chrome:**
1. Click the lock icon (🔒) in address bar
2. Click "Site settings"
3. Set "Microphone" to "Allow"

### **Firefox:**
1. Click the shield icon in address bar
2. Click "Site Permissions"
3. Set "Microphone" to "Allow"

### **Mobile Browsers:**
- Usually prompt automatically for microphone access
- Look for microphone icon in address bar
- Tap to allow access

## 📊 **Server Information**
- **Host**: 10.80.2.40
- **Main Port**: 5000 (Flask app)
- **Test Port**: 8081 (Simple HTTP server)
- **Status**: Both servers running and accessible

## 🛠 **Server Commands**

### **Start Main Server:**
```bash
cd /home/novel/voice_to_voice
source voice_env/bin/activate
PYTHONPATH=/home/novel/voice_to_voice python src/web/app.py
```

### **Start Test Server:**
```bash
cd /home/novel/voice_to_voice
python3 -m http.server 8081
```

### **Check Server Status:**
```bash
curl http://10.80.2.40:5000/api/status
``` 