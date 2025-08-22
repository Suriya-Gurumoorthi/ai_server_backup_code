// WebRTC Interface for Real-time Voice Communication
class WebRTCVoiceInterface {
    constructor() {
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.dataChannel = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.isConnected = false;
        this.isRecording = false;
        this.audioChunks = [];
        this.silenceTimer = null;
        this.silenceThreshold = 1500; // 1.5 seconds of silence
        this.lastAudioTime = 0;
        
        // Audio processing
        this.audioProcessor = null;
        this.audioBuffer = [];
        this.bufferSize = 4096;
        this.sampleRate = 16000;
        
        // Conversation state management
        this.isUserSpeaking = false;
        this.isAIResponding = false;
        this.currentAudioElement = null;
        this.pendingAudioChunks = [];
        this.conversationHistory = [];
        
        // WebSocket for signaling
        this.signalingSocket = null;
        this.sessionId = null;
        
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        this.webrtcStatus = document.getElementById('webrtcStatus');
        this.connectBtn = document.getElementById('connectWebRTC');
        this.disconnectBtn = document.getElementById('disconnectWebRTC');
        this.audioVisualizer = document.getElementById('audioVisualizer');
        this.webrtcMessages = document.getElementById('webrtcMessages');
    }

    attachEventListeners() {
        if (this.connectBtn) {
            this.connectBtn.addEventListener('click', () => this.connect());
        }
        if (this.disconnectBtn) {
            this.disconnectBtn.addEventListener('click', () => this.disconnect());
        }
    }

    async connect() {
        try {
            this.updateStatus('Connecting...', 'connecting');
            
            // Initialize WebSocket for signaling
            await this.initializeSignaling();
            
            // Get user media
            this.localStream = await this.getUserMedia();
            
            // Skip WebRTC peer connection for now - just start audio processing directly
            console.log('Starting direct audio processing...');
            this.startAudioProcessing();
            
            // Update status
            this.updateStatus('Connected - Listening', 'connected');
            this.isConnected = true;

        } catch (error) {
            console.error('Connection failed:', error);
            this.updateStatus('Connection failed', 'error');
        }
    }

    async initializeSignaling() {
        return new Promise((resolve, reject) => {
            // Use Socket.IO instead of raw WebSocket
            this.signalingSocket = io();
            
            this.signalingSocket.on('connect', () => {
                console.log('Signaling connection established');
                this.sessionId = 'session_' + Date.now();
                this.signalingSocket.emit('join', {
                    sessionId: this.sessionId
                });
                resolve();
            });
            
            this.signalingSocket.on('joined', (data) => {
                console.log('Joined session:', data);
            });
            
            this.signalingSocket.on('answer', (data) => {
                console.log('Received answer data:', data);
                // Not using WebRTC peer connection for now
            });
            
            this.signalingSocket.on('ice-candidate', (data) => {
                console.log('Received ICE candidate:', data);
                // Not using WebRTC peer connection for now
            });
            
            this.signalingSocket.on('ai-response', (data) => {
                this.handleSignalingMessage({type: 'ai-response', ...data});
            });
            
            this.signalingSocket.on('connect_error', (error) => {
                console.error('Signaling error:', error);
                reject(error);
            });
        });
    }

    handleSignalingMessage(message) {
        console.log('Handling signaling message:', message);
        
        switch (message.type) {
            case 'answer':
                try {
                    // Create proper RTCSessionDescriptionInit object
                    const sessionDescription = {
                        type: 'answer',
                        sdp: message.sdp
                    };
                    console.log('Setting remote description:', sessionDescription);
                    this.peerConnection.setRemoteDescription(new RTCSessionDescription(sessionDescription));
                } catch (error) {
                    console.error('Error setting remote description:', error);
                    console.error('Message received:', message);
                }
                break;
            case 'ice-candidate':
                try {
                    this.peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                } catch (error) {
                    console.error('Error adding ICE candidate:', error);
                }
                break;
            case 'ai-response':
                this.handleAIResponse(message);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    async getUserMedia() {
        const constraints = {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: this.sampleRate,
                channelCount: 1
            }
        };
        
        return await navigator.mediaDevices.getUserMedia(constraints);
    }

    startAudioProcessing() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = this.audioContext.createMediaStreamSource(this.localStream);
        
        // Create audio processor for real-time processing
        this.audioProcessor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
        
        this.audioProcessor.onaudioprocess = (event) => {
            const inputBuffer = event.inputBuffer;
            const inputData = inputBuffer.getChannelData(0);
            
            // Check for silence
            const isSilent = this.checkSilence(inputData);
            
            if (!isSilent) {
                // User is speaking
                if (!this.isUserSpeaking) {
                    this.isUserSpeaking = true;
                    this.stopAIResponse(); // Stop AI if it's talking
                    this.updateStatus('User Speaking...', 'speaking');
                }
                
                this.lastAudioTime = Date.now();
                this.audioBuffer.push(...Array.from(inputData));
                
                // Process audio in chunks
                if (this.audioBuffer.length >= this.bufferSize * 10) { // ~1 second of audio
                    this.processAudioChunk();
                }
            } else {
                // Check if we've been silent for too long
                if (Date.now() - this.lastAudioTime > this.silenceThreshold && this.isUserSpeaking) {
                    this.isUserSpeaking = false;
                    this.updateStatus('Processing...', 'processing');
                    this.processAudioChunk(); // Process remaining audio
                }
            }
        };
        
        source.connect(this.audioProcessor);
        this.audioProcessor.connect(this.audioContext.destination);
        
        this.isRecording = true;
        this.updateStatus('Connected - Listening', 'connected');
    }

    checkSilence(audioData) {
        // Improved silence detection with better threshold
        const threshold = 0.015; // Increased threshold to reduce background noise
        const rms = Math.sqrt(audioData.reduce((sum, sample) => sum + sample * sample, 0) / audioData.length);
        return rms < threshold;
    }

    async processAudioChunk() {
        if (this.audioBuffer.length === 0) return;
        
        // Don't process if AI is responding
        if (this.isAIResponding) {
            console.log('AI is responding, skipping audio chunk');
            this.audioBuffer = []; // Clear buffer anyway
            return;
        }
        
        // Convert audio buffer to WAV format
        const audioBlob = this.convertToWAV(this.audioBuffer);
        this.audioBuffer = []; // Clear buffer
        
        // Send to server for processing
        await this.sendAudioToServer(audioBlob);
    }

    convertToWAV(audioData) {
        // Simple WAV conversion (you might want to use a library like wav-encoder)
        const buffer = new ArrayBuffer(44 + audioData.length * 2);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + audioData.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, this.sampleRate, true);
        view.setUint32(28, this.sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, audioData.length * 2, true);
        
        // Audio data
        for (let i = 0; i < audioData.length; i++) {
            view.setInt16(44 + i * 2, audioData[i] * 0x7FFF, true);
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }

    async sendAudioToServer(audioBlob) {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'realtime.wav');
            formData.append('sessionId', this.sessionId);
            
            const response = await fetch('/api/process_realtime_audio', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Add user message to conversation history if text was recognized
                if (result.text && result.text.trim()) {
                    this.addMessage('User', result.text);
                    this.conversationHistory.push({ speaker: 'User', text: result.text, timestamp: Date.now() });
                }
                
                this.handleAIResponse(result);
            }
        } catch (error) {
            console.error('Error sending audio to server:', error);
            this.updateStatus('Connected - Listening', 'connected');
        }
    }

    handleAIResponse(response) {
        if (response.text) {
            // Add AI message to conversation history
            this.addMessage('AI', response.text);
            this.conversationHistory.push({ speaker: 'AI', text: response.text, timestamp: Date.now() });
        }
        
        if (response.audio_url) {
            this.playAIResponse(response.audio_url);
        }
    }

    async playAIResponse(audioUrl) {
        try {
            // Stop any currently playing audio
            this.stopAIResponse();
            
            // Set AI responding state
            this.isAIResponding = true;
            this.updateStatus('AI Speaking...', 'ai-speaking');
            
            const audio = new Audio(audioUrl);
            this.currentAudioElement = audio;
            
            // Handle audio completion
            audio.onended = () => {
                this.isAIResponding = false;
                this.currentAudioElement = null;
                this.updateStatus('Connected - Listening', 'connected');
            };
            
            // Handle audio errors
            audio.onerror = () => {
                this.isAIResponding = false;
                this.currentAudioElement = null;
                this.updateStatus('Connected - Listening', 'connected');
                console.error('Error playing AI response');
            };
            
            await audio.play();
        } catch (error) {
            console.error('Error playing AI response:', error);
            this.isAIResponding = false;
            this.currentAudioElement = null;
            this.updateStatus('Connected - Listening', 'connected');
        }
    }
    
    stopAIResponse() {
        if (this.currentAudioElement) {
            this.currentAudioElement.pause();
            this.currentAudioElement.currentTime = 0;
            this.currentAudioElement = null;
        }
        this.isAIResponding = false;
    }

    playRemoteAudio() {
        if (this.remoteStream) {
            const audio = new Audio();
            audio.srcObject = this.remoteStream;
            audio.play();
        }
    }

    handleConnectionStateChange() {
        const state = this.peerConnection.connectionState;
        console.log('Connection state:', state);
        
        switch (state) {
            case 'connected':
                this.isConnected = true;
                this.updateStatus('Connected', 'connected');
                break;
            case 'disconnected':
            case 'failed':
                this.isConnected = false;
                this.updateStatus('Disconnected', 'disconnected');
                break;
        }
    }

    disconnect() {
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        if (this.signalingSocket) {
            this.signalingSocket.disconnect();
            this.signalingSocket = null;
        }
        
        this.isConnected = false;
        this.isRecording = false;
        this.updateStatus('Disconnected', 'disconnected');
    }

    updateStatus(status, type) {
        if (this.webrtcStatus) {
            this.webrtcStatus.textContent = status;
            this.webrtcStatus.className = `status ${type}`;
        }
        console.log(`Status: ${status} (${type})`);
    }

    addMessage(speaker, text) {
        if (this.webrtcMessages) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${speaker.toLowerCase()}`;
            messageDiv.innerHTML = `<strong>${speaker}:</strong> ${text}`;
            this.webrtcMessages.appendChild(messageDiv);
            this.webrtcMessages.scrollTop = this.webrtcMessages.scrollHeight;
        }
    }
}

// Initialize WebRTC interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.webrtcInterface = new WebRTCVoiceInterface();
}); 