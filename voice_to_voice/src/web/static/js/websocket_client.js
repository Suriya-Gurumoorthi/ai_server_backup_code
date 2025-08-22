// WebSocket Client for real-time communication
class WebSocketClient {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    connect() {
        try {
            // Check if WebSocket is supported
            if (!window.WebSocket) {
                console.warn('WebSocket not supported in this browser');
                return;
            }
            
            // Only connect if we're on localhost or have WebSocket support
            const isLocalhost = window.location.hostname === 'localhost' || 
                               window.location.hostname === '127.0.0.1';
            
            if (!isLocalhost) {
                console.log('WebSocket disabled for non-localhost connections');
                return;
            }
            
            this.socket = new WebSocket(`ws://${window.location.host}/ws`);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection failed:', error);
        }
    }

    setupEventHandlers() {
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.socket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.attemptReconnect();
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleMessage(data) {
        switch (data.type) {
            case 'status':
                this.updateStatus(data.message);
                break;
            case 'conversation':
                this.updateConversation(data.message);
                break;
            case 'audio':
                this.playAudio(data.audio_url);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    sendMessage(type, data) {
        if (this.isConnected) {
            this.socket.send(JSON.stringify({
                type: type,
                data: data
            }));
        } else {
            console.warn('WebSocket not connected');
        }
    }

    updateStatus(message) {
        const statusElement = document.getElementById('statusDisplay');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }

    updateConversation(message) {
        const conversationElement = document.getElementById('conversationLog');
        if (conversationElement) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.innerHTML = message;
            conversationElement.appendChild(messageDiv);
            conversationElement.scrollTop = conversationElement.scrollHeight;
        }
    }

    playAudio(audioUrl) {
        const audioElement = document.getElementById('audioPlayer');
        if (audioElement) {
            audioElement.src = audioUrl;
            audioElement.play();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.log('WebSocket connection not available - using HTTP fallback');
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }
}

// Initialize WebSocket client when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.wsClient = new WebSocketClient();
    window.wsClient.connect();
});
