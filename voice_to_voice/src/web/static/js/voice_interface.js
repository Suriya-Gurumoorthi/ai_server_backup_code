// Voice Interface JavaScript

// Browser compatibility check
function checkBrowserCompatibility() {
    const issues = [];
    
    // Check if we're on HTTPS or localhost or your network IP
    const isSecure = window.location.protocol === 'https:' || 
                    window.location.hostname === 'localhost' || 
                    window.location.hostname === '127.0.0.1' ||
                    window.location.hostname === '10.80.2.40'; // Allow your network IP
    
    // For network access, we'll be more permissive but show a warning
    if (!isSecure) {
        issues.push('Not on HTTPS or localhost - microphone access may be blocked');
    }
    
    // Enhanced getUserMedia support check
    const hasGetUserMedia = !!(navigator.mediaDevices?.getUserMedia || 
                              navigator.getUserMedia ||
                              navigator.webkitGetUserMedia || 
                              navigator.mozGetUserMedia ||
                              navigator.msGetUserMedia);
    
    if (!hasGetUserMedia) {
        issues.push('getUserMedia not supported');
    }
    
    // Debug information
    console.log('Browser compatibility check:');
    console.log('- Protocol:', window.location.protocol);
    console.log('- Hostname:', window.location.hostname);
    console.log('- Is secure:', isSecure);
    console.log('- navigator.mediaDevices:', !!navigator.mediaDevices);
    console.log('- navigator.mediaDevices.getUserMedia:', !!navigator.mediaDevices?.getUserMedia);
    console.log('- navigator.getUserMedia:', !!navigator.getUserMedia);
    console.log('- navigator.webkitGetUserMedia:', !!navigator.webkitGetUserMedia);
    console.log('- navigator.mozGetUserMedia:', !!navigator.mozGetUserMedia);
    console.log('- navigator.msGetUserMedia:', !!navigator.msGetUserMedia);
    console.log('- Has getUserMedia:', hasGetUserMedia);
    
    // Special warning for network IP access
    if (window.location.hostname === '10.80.2.40' && window.location.protocol !== 'https:') {
        console.warn('⚠️ Accessing via network IP without HTTPS. Microphone access may be blocked by browser security policies.');
        console.warn('💡 Recommendation: Use localhost (127.0.0.1:5000) for testing, or set up HTTPS for network access.');
    }
    
    if (issues.length > 0) {
        console.warn('Browser compatibility issues:', issues);
        return false;
    }
    
    return true;
}

class NovelOfficeInterface {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.currentAudio = null;
        this.isMuted = false;
        this.currentRole = 'fahad_sales';
        this.isVoiceInput = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.startConversation();
        this.initializeFAB();
    }

    initializeElements() {
        // Recording controls
        this.startRecordingBtn = document.getElementById('startRecording');
        this.stopRecordingBtn = document.getElementById('stopRecording');
        this.interruptBtn = document.getElementById('interruptAI');
        
        // Status elements
        this.recordingStatus = document.getElementById('recordingStatus');
        this.aiSpeakingStatus = document.getElementById('aiSpeakingStatus');
        this.memoryStatus = document.getElementById('memoryStatus');
        this.connectionStatus = document.getElementById('connectionStatus');
        
        // Conversation elements
        this.messagesDiv = document.getElementById('messages');
        this.textInput = document.getElementById('textInput');
        this.sendTextBtn = document.getElementById('sendText');
        
        // Control buttons
        this.clearMemoryBtn = document.getElementById('clearMemory');
        this.toggleMuteBtn = document.getElementById('toggleMute');
        
        // FAB elements
        this.fabMain = document.getElementById('fabMain');
        this.fabOptions = document.getElementById('fabOptions');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingMessage = document.getElementById('loadingMessage');
    }

    attachEventListeners() {
        // Recording controls
        this.startRecordingBtn.addEventListener('click', () => this.startRecording());
        this.stopRecordingBtn.addEventListener('click', () => this.stopRecording());
        this.interruptBtn.addEventListener('click', () => this.interruptAI());
        
        // Text input
        this.sendTextBtn.addEventListener('click', () => this.sendTextMessage());
        this.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextMessage();
            }
        });
        
        // Control buttons
        this.clearMemoryBtn.addEventListener('click', () => this.clearMemory());
        this.toggleMuteBtn.addEventListener('click', () => this.toggleMute());
        
        // Quick action buttons
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleQuickAction(action);
            });
        });
        
        // FAB
        this.fabMain.addEventListener('click', () => this.toggleFAB());
        document.querySelectorAll('.fab-option').forEach(option => {
            option.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleFABAction(action);
            });
        });
    }

    initializeFAB() {
        // Close FAB when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.fabMain.contains(e.target) && !this.fabOptions.contains(e.target)) {
                this.fabOptions.classList.remove('show');
            }
        });
    }

    toggleFAB() {
        this.fabOptions.classList.toggle('show');
    }

    handleQuickAction(action) {
        const messages = {
            'office-space': "I'm looking for office space for my business. Can you tell me about your flexible office solutions?",
            'investment': "I'm interested in your real estate investment opportunities. How does the investment program work?",
            'amenities': "What amenities do you provide with your office spaces? I'd like to know about internet, meeting rooms, and other facilities."
        };
        
        if (messages[action]) {
            this.textInput.value = messages[action];
            this.sendTextMessage();
        }
    }

    handleFABAction(action) {
        switch (action) {
            case 'test-microphone':
                this.testMicrophone();
                break;
            case 'office-info':
                this.showOfficeInfo();
                break;
            case 'contact-info':
                this.showContactInfo();
                break;
        }
        this.toggleFAB();
    }

    showOfficeInfo() {
        this.addMessage('system', 'Novel Office provides flexible, fully equipped office spaces in Bangalore, India. We offer customized solutions for startups, SMEs, and corporates with amenities like high-speed internet, meeting rooms, 24/7 access, and community programs.');
    }

    showContactInfo() {
        this.addMessage('system', 'You can reach Novel Office through our website or contact our sales team directly. I\'m here to help you with any inquiries about our office spaces and investment opportunities.');
    }

    async startConversation() {
        try {
            // Clear any existing memory first
            await fetch('/api/clear_memory', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ role: this.currentRole })
            });

            // Set role to Fahad
            await fetch('/api/change_role', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ role: this.currentRole })
            });

            this.updateConnectionStatus('Connected', 'success');
            this.updateMemoryStatus();
            
        } catch (error) {
            console.error('Error starting conversation:', error);
            this.updateConnectionStatus('Connection Error', 'error');
        }
    }

    async startRecording() {
        try {
            this.showLoading('Requesting microphone access...');
            
            // Stop any current AI speech
            this.stopAISpeech();
            
            const stream = await this.getUserMediaWithTimeout({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            this.updateRecordingStatus('Recording...', 'recording');
            this.startRecordingBtn.disabled = true;
            this.stopRecordingBtn.disabled = false;
            this.interruptBtn.disabled = true;
            
            // Add recording animation
            document.body.classList.add('recording');
            
            this.hideLoading();
            
        } catch (error) {
            this.hideLoading();
            this.showError('Failed to start recording: ' + error.message);
            console.error('Recording error:', error);
        }
    }

    async stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            this.updateRecordingStatus('Processing...', 'processing');
            this.startRecordingBtn.disabled = false;
            this.stopRecordingBtn.disabled = true;
            
            // Remove recording animation
            document.body.classList.remove('recording');
            
            // Stop all tracks
            if (this.mediaRecorder.stream) {
                this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
    }

    async processRecording() {
        try {
            this.showLoading('Processing your voice...');
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('role', this.currentRole);
            
            // Set flag to indicate this is voice input
            this.isVoiceInput = true;
            
            const response = await fetch('/api/process_audio', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.text && result.response) {
                this.handleResponse(result);
            } else {
                throw new Error(result.error || 'Failed to process audio');
            }
            
        } catch (error) {
            this.showError('Error processing audio: ' + error.message);
            console.error('Audio processing error:', error);
        } finally {
            this.hideLoading();
            this.updateRecordingStatus('Ready', 'ready');
        }
    }

    async sendTextMessage() {
        const text = this.textInput.value.trim();
        if (!text) return;

        try {
            this.showLoading('Sending message...');
            
            // Set flag to indicate this is text input
            this.isVoiceInput = false;
            
            // Add user message to conversation
            this.addMessage('user', text);
            this.textInput.value = '';

            const response = await fetch('/api/process_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const result = await response.json();
            
            if (result.response) {
                this.handleResponse(result);
            } else {
                throw new Error(result.error || 'Failed to process message');
            }
            
        } catch (error) {
            this.showError('Error sending message: ' + error.message);
            console.error('Text message error:', error);
        } finally {
            this.hideLoading();
        }
    }

    handleResponse(result) {
        // Add user message for voice input only (text input already adds it)
        if (result.text && this.isVoiceInput) {
            this.addMessage('user', result.text);
        }
        
        // Add assistant response
        const assistantMessage = this.addMessage('assistant', result.response);
        
        // Play audio if available and not muted
        if (result.audio_url && !this.isMuted) {
            this.playAudio(result.audio_url, assistantMessage);
        }
        
        // Update memory status
        this.updateMemoryStatus();
    }

    async playAudio(audioUrl, messageElement = null) {
        try {
            // Stop any current audio
            this.stopAISpeech();
            
            this.updateAISpeakingStatus('Speaking...', 'speaking');
            document.body.classList.add('ai-speaking');
            
            // Add visual indicator to the message if provided
            if (messageElement) {
                messageElement.classList.add('playing-audio');
            }
            
            const audio = new Audio(audioUrl);
            this.currentAudio = audio;
            
            audio.onended = () => {
                this.updateAISpeakingStatus('Silent', 'silent');
                document.body.classList.remove('ai-speaking');
                if (messageElement) {
                    messageElement.classList.remove('playing-audio');
                }
                this.currentAudio = null;
            };
            
            audio.onerror = () => {
                this.updateAISpeakingStatus('Error', 'error');
                document.body.classList.remove('ai-speaking');
                if (messageElement) {
                    messageElement.classList.remove('playing-audio');
                }
                this.currentAudio = null;
            };
            
            await audio.play();
            
        } catch (error) {
            console.error('Audio playback error:', error);
            this.updateAISpeakingStatus('Error', 'error');
            document.body.classList.remove('ai-speaking');
            if (messageElement) {
                messageElement.classList.remove('playing-audio');
            }
        }
    }

    stopAISpeech() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }
        this.updateAISpeakingStatus('Interrupted', 'interrupted');
        document.body.classList.remove('ai-speaking');
    }

    interruptAI() {
        this.stopAISpeech();
        this.interruptBtn.disabled = true;
        
        // Re-enable interrupt button after a short delay
        setTimeout(() => {
            this.interruptBtn.disabled = false;
        }, 2000);
    }

    async clearMemory() {
        try {
            this.showLoading('Clearing conversation...');
            
            const response = await fetch('/api/clear_memory', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ role: this.currentRole })
            });

            const result = await response.json();
            
            if (result.success) {
                this.memoryStatus.textContent = result.memory_summary;
                this.addMessage('system', 'Conversation memory cleared. How can I help you today?');
            } else {
                throw new Error(result.error || 'Failed to clear memory');
            }
            
        } catch (error) {
            this.showError('Error clearing memory: ' + error.message);
            console.error('Clear memory error:', error);
        } finally {
            this.hideLoading();
        }
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        const icon = this.toggleMuteBtn.querySelector('i');
        const title = this.toggleMuteBtn.getAttribute('title');
        
        if (this.isMuted) {
            icon.className = 'fas fa-volume-mute';
            this.toggleMuteBtn.setAttribute('title', 'Unmute');
        } else {
            icon.className = 'fas fa-volume-up';
            this.toggleMuteBtn.setAttribute('title', 'Mute');
        }
    }

    async testMicrophone() {
        try {
            this.showLoading('Testing microphone...');
            
            const stream = await this.getUserMediaWithTimeout({ audio: true });
            
            // Stop the stream immediately
            stream.getTracks().forEach(track => track.stop());
            
            this.addMessage('system', 'Microphone test successful! Your microphone is working properly.');
            
        } catch (error) {
            this.showError('Microphone test failed: ' + error.message);
            console.error('Microphone test error:', error);
        } finally {
            this.hideLoading();
        }
    }

    getUserMediaWithTimeout(constraints, timeoutMs = 10000) {
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error('Microphone access request timed out. Please check your browser permissions.'));
            }, timeoutMs);
            
            navigator.mediaDevices.getUserMedia(constraints)
                .then(stream => {
                    clearTimeout(timeoutId);
                    resolve(stream);
                })
                .catch(error => {
                    clearTimeout(timeoutId);
                    reject(error);
                });
        });
    }

    addMessage(speaker, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${speaker}-message`;
        
        const speakerLabel = speaker === 'user' ? 'You' : 
                           speaker === 'assistant' ? 'Fahad' : 
                           speaker === 'system' ? 'System' : speaker;
        
        // Get appropriate avatar icon
        let avatarIcon = 'fas fa-user';
        if (speaker === 'assistant') {
            avatarIcon = 'fas fa-user-tie';
        } else if (speaker === 'system') {
            avatarIcon = 'fas fa-info-circle';
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <strong>${speakerLabel}</strong>
                    <span class="message-time">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="message-text">${text}</div>
            </div>
        `;
        
        this.messagesDiv.appendChild(messageDiv);
        this.messagesDiv.scrollTop = this.messagesDiv.scrollHeight;
        
        // Add fade-in animation
        messageDiv.style.opacity = '0';
        setTimeout(() => {
            messageDiv.style.transition = 'opacity 0.3s ease-in';
            messageDiv.style.opacity = '1';
        }, 10);
        
        return messageDiv;
    }

    updateRecordingStatus(status, type) {
        this.recordingStatus.innerHTML = `<i class="fas fa-circle"></i> ${status}`;
        this.recordingStatus.className = `status-value ${type}`;
    }

    updateAISpeakingStatus(status, type) {
        this.aiSpeakingStatus.innerHTML = `<i class="fas fa-circle"></i> ${status}`;
        this.aiSpeakingStatus.className = `status-value ${type}`;
    }

    updateConnectionStatus(status, type) {
        this.connectionStatus.innerHTML = `<i class="fas fa-circle"></i> ${status}`;
        this.connectionStatus.className = `status-indicator ${type}`;
    }

    async updateMemoryStatus() {
        try {
            const response = await fetch('/api/memory_summary?role=' + this.currentRole);
            const result = await response.json();
            
            if (result.success) {
                this.memoryStatus.textContent = result.memory_summary;
            }
        } catch (error) {
            console.error('Error updating memory status:', error);
        }
    }

    showLoading(message) {
        this.loadingMessage.textContent = message;
        this.loadingOverlay.classList.add('show');
    }

    hideLoading() {
        this.loadingOverlay.classList.remove('show');
    }

    showError(message) {
        this.addMessage('system', `Error: ${message}`);
    }
}

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new NovelOfficeInterface();
});
