# 🚀 Quick Start Guide - AI Interview Evaluation API

## ✅ **API is Now Running Successfully!**

Your AI Interview Evaluation API is ready to use. Here's how to get started:

## 🌐 **Access Your API**

### **Web Interface**
- **URL**: http://localhost:8000
- **Features**: Beautiful drag-and-drop interface for uploading audio files
- **Real-time**: Live processing status updates

### **API Documentation**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### **Health Check**
- **URL**: http://localhost:8000/health
- **Returns**: API status and model loading state

## 📱 **How to Use the Web Interface**

1. **Open your browser** and go to http://localhost:8000
2. **Upload an audio file** by:
   - Dragging and dropping a file onto the upload area
   - Clicking the upload area to browse for a file
3. **Supported formats**: WAV, MP3, FLAC, M4A (Max 50MB)
4. **Watch the processing** in real-time
5. **View results** in a beautiful, professional format
6. **Download the report** as needed

## 🔧 **API Endpoints**

### **Upload Audio File**
```bash
POST /upload
Content-Type: multipart/form-data
```

### **Check Processing Status**
```bash
GET /status/{job_id}
```

### **Get Results**
```bash
GET /results/{job_id}
```

### **Delete Job**
```bash
DELETE /jobs/{job_id}
```

## 💻 **Command Line Usage**

### **Test the API**
```bash
python test_api.py
```

### **Use the API Client**
```bash
python api_client_example.py
```

### **CLI Processing (Alternative)**
```bash
python main.py Audios/your_audio.wav
```

## 🛠️ **Server Management**

### **Start the API Server**
```bash
# Activate virtual environment
source /home/novel/venv/bin/activate

# Start the server
python start_api.py
```

### **Stop the Server**
Press `Ctrl+C` in the terminal where the server is running.

### **Check Server Status**
```bash
curl http://localhost:8000/health
```

## 📊 **What You Get**

The system evaluates candidates on:
- **Confidence** (tone, hesitation, filler words)
- **Pronunciation & Clarity**
- **Fluency & Logical Coherence**
- **Emotional Tone** (nervous, calm, assertive)
- **Grammar & Vocabulary Usage**
- **Overall Suitability Score** (0-100)
- **Final Decision** (Hire/Reject)

## 🔒 **Important Notes**

- **Model Loading**: The Ultravox model loads automatically when first needed
- **File Cleanup**: Uploaded files are automatically cleaned up after processing
- **Background Processing**: Uploads don't block the interface
- **Real-time Updates**: Status updates every 2 seconds during processing

## 🎯 **Next Steps**

1. **Test with your audio files** by uploading them through the web interface
2. **Explore the API documentation** at http://localhost:8000/docs
3. **Integrate with your applications** using the REST API endpoints
4. **Customize the evaluation criteria** by editing `configs/config.py`

---

**🎉 Congratulations! Your AI Interview Evaluation API is ready for production use!** 