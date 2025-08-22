# 🌐 Accessing AI Interview API from Your PC

## **Current Setup:**
- **Server**: Linux machine running the API (IP: 10.80.2.40)
- **Your PC**: Windows/Mac with Chrome browser
- **Goal**: Access the web interface from your PC

## **Method 1: Direct Network Access**

### **Step 1: Check Network Connectivity**
From your PC, open Command Prompt (Windows) or Terminal (Mac) and run:
```bash
ping 10.80.2.40
```

### **Step 2: Access the Web Interface**
If ping works, open Chrome and go to:
```
http://10.80.2.40:8000
```

### **Step 3: Test API Endpoints**
- **Web Interface**: http://10.80.2.40:8000
- **API Docs**: http://10.80.2.40:8000/docs
- **Health Check**: http://10.80.2.40:8000/health

## **Method 2: SSH Tunnel (If Method 1 fails)**

### **Step 1: Create SSH Tunnel**
From your PC, open terminal/command prompt and run:
```bash
ssh -L 8000:localhost:8000 username@10.80.2.40
```

Replace `username` with your actual username on the server.

### **Step 2: Access via Localhost**
Once the tunnel is established, open Chrome and go to:
```
http://localhost:8000
```

## **Method 3: Port Forwarding (Alternative)**

### **Step 1: Configure Port Forwarding**
If you're using a cloud server or VM, configure port forwarding:
- **External Port**: 8000
- **Internal IP**: 10.80.2.40
- **Internal Port**: 8000

### **Step 2: Access via External IP**
Use the external IP address:
```
http://[EXTERNAL_IP]:8000
```

## **Troubleshooting**

### **If "Connection Refused":**
1. Check if the server is running:
   ```bash
   # On the server
   curl http://localhost:8000/health
   ```

2. Check firewall settings:
   ```bash
   # On the server
   sudo ufw status
   sudo ufw allow 8000
   ```

### **If "Page Not Found":**
1. Verify the server is running on the correct port:
   ```bash
   # On the server
   netstat -tlnp | grep :8000
   ```

2. Check if the API is accessible locally:
   ```bash
   # On the server
   curl http://localhost:8000/
   ```

## **Quick Test Commands**

### **From Your PC:**
```bash
# Test connectivity
ping 10.80.2.40

# Test API health
curl http://10.80.2.40:8000/health

# Test web interface
curl http://10.80.2.40:8000/
```

### **From the Server:**
```bash
# Check if server is running
ps aux | grep python

# Check port usage
netstat -tlnp | grep :8000

# Test local access
curl http://localhost:8000/health
```

## **Security Notes**

- The API currently allows CORS from all origins
- In production, configure proper firewall rules
- Consider using HTTPS for secure connections
- Restrict access to trusted IP addresses if needed

## **Next Steps**

1. **Try Method 1** (Direct Network Access) first
2. **If it doesn't work**, try Method 2 (SSH Tunnel)
3. **Test the web interface** by uploading an audio file
4. **Check the API documentation** at /docs endpoint

---

**Need Help?** If none of these methods work, please provide:
- Your PC's operating system (Windows/Mac/Linux)
- Network setup (same network/VPN/cloud server)
- Any error messages you see 