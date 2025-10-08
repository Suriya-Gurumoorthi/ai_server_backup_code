#!/bin/bash

# Ultravox Server Setup Script
# This script sets up the complete environment for running the Ultravox server
# Author: Auto-generated setup script
# Date: $(date)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging setup
LOG_FILE="install/setup.log"
mkdir -p install
exec > >(tee -a "$LOG_FILE") 2>&1

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check NVIDIA GPU
check_nvidia_gpu() {
    print_status "Checking for NVIDIA GPU..."
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        return 0
    else
        print_warning "NVIDIA GPU not detected or nvidia-smi not available"
        return 1
    fi
}

# Function to check CUDA installation
check_cuda() {
    print_status "Checking CUDA installation..."
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        print_success "CUDA is installed: Version $CUDA_VERSION"
        return 0
    else
        print_warning "CUDA not found"
        return 1
    fi
}

# Function to install NVIDIA drivers and CUDA
install_nvidia_cuda() {
    print_status "Installing NVIDIA drivers and CUDA toolkit..."
    
    # Update package list
    sudo apt update
    
    # Install NVIDIA drivers (latest stable)
    print_status "Installing NVIDIA drivers..."
    sudo apt install -y nvidia-driver-535
    
    # Install CUDA toolkit (version 12.1 for compatibility with PyTorch)
    print_status "Installing CUDA toolkit 12.1..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-1
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    
    print_success "NVIDIA drivers and CUDA toolkit installed"
    print_warning "Please reboot your system to complete NVIDIA driver installation"
}

# Function to validate CUDA in Python
validate_cuda_python() {
    print_status "Validating CUDA in Python..."
    
    # Create a temporary validation script
    cat > temp_cuda_validation.py << 'EOF'
import torch
import sys

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print("✅ CUDA validation successful")
else:
    print("⚠️  CUDA is not available. PyTorch will use CPU.")
    print("This is normal if you don't have an NVIDIA GPU or CUDA is not properly installed.")
EOF

    # Use python3 from the virtual environment
    python3 temp_cuda_validation.py
    rm temp_cuda_validation.py
    
    if [ $? -eq 0 ]; then
        print_success "CUDA validation completed"
    else
        print_error "CUDA validation failed"
        exit 1
    fi
}

# Function to setup Hugging Face authentication
setup_huggingface_auth() {
    print_status "Setting up Hugging Face authentication..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env file not found. Please copy .env.example to .env and add your HF_TOKEN"
            print_warning "You can get your token from: https://huggingface.co/settings/tokens"
            cp .env.example .env
        else
            print_error ".env.example file not found. Please create .env file with HF_TOKEN"
            exit 1
        fi
    fi
    
    # Load .env file and check for HF_TOKEN
    if [ -f ".env" ]; then
        source .env
        if [ -z "$HF_TOKEN" ]; then
            print_error "HF_TOKEN not found in .env file"
            print_warning "Please add your Hugging Face token to the .env file"
            exit 1
        else
            print_success "Hugging Face token found in .env file"
        fi
    fi
}

# Main setup function
main() {
    print_status "Starting Ultravox Server Setup..."
    print_status "Log file: $LOG_FILE"
    
    # Step 1: Update APT and Install Essentials
    print_status "Step 1: Updating APT and installing essentials..."
    sudo apt update
    sudo apt install -y \
        wget \
        curl \
        git \
        build-essential \
        software-properties-common \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        portaudio19-dev \
        python3-pyaudio \
        onnxruntime
    
    print_success "Step 1 completed: APT updated and essentials installed"
    
    # Step 2: Check for NVIDIA GPU
    print_status "Step 2: Checking for NVIDIA GPU..."
    if check_nvidia_gpu; then
        print_success "Step 2 completed: NVIDIA GPU detected"
    else
        print_warning "Step 2: No NVIDIA GPU detected - will use CPU"
    fi
    
    # Step 3: Install Python 3.11 (already done in step 1)
    print_status "Step 3: Python 3.11 installation verified..."
    python3.11 --version
    print_success "Step 3 completed: Python 3.11 is available"
    
    # Step 4: Install NVIDIA Drivers and CUDA Toolkit
    print_status "Step 4: Checking NVIDIA drivers and CUDA..."
    if check_nvidia_gpu && check_cuda; then
        print_success "Step 4 completed: NVIDIA drivers and CUDA already installed"
    else
        if check_nvidia_gpu; then
            print_status "Installing CUDA toolkit..."
            install_nvidia_cuda
        else
            print_warning "Step 4: No NVIDIA GPU detected - skipping CUDA installation"
        fi
    fi
    
    # Step 5: Create and Activate Python Virtual Environment
    print_status "Step 5: Creating and activating Python virtual environment..."
    python3.11 -m venv venv
    source venv/bin/activate
    
    # Create necessary directories
    print_status "Creating required directories..."
    mkdir -p conversation_history prompt_logs captured_audio models
    print_success "Step 5 completed: Virtual environment created and activated"
    
    # Step 6: Install all packages from requirements.txt
    print_status "Step 6: Installing packages from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
        print_success "Step 6 completed: All packages installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Step 7: CUDA Validation in Python (after packages are installed)
    print_status "Step 7: Validating CUDA in Python..."
    if check_nvidia_gpu; then
        validate_cuda_python
        print_success "Step 7 completed: CUDA validation successful"
    else
        print_warning "Step 7: Skipping CUDA validation (no GPU detected)"
    fi
    
    # Step 8: Hugging Face Authentication for Gated Llama
    print_status "Step 8: Setting up Hugging Face authentication..."
    setup_huggingface_auth
    print_success "Step 8 completed: Hugging Face authentication configured"
    
    # Step 8.5: Download Piper TTS Model Files
    print_status "Step 8.5: Downloading Piper TTS model files..."
    if [ ! -f "models/en_US-lessac-medium.onnx" ] || [ ! -f "models/en_US-lessac-medium.json" ]; then
        print_status "Downloading Piper TTS model files..."
        cd models
        wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
        wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.json
        cd ..
        print_success "Piper TTS model files downloaded successfully"
    else
        print_success "Piper TTS model files already exist"
    fi
    print_success "Step 8.5 completed: Piper TTS models ready"
    
    # Step 8.6: Install Cursor IDE
    print_status "Step 8.6: Installing Cursor IDE..."
    if ! command_exists cursor; then
        print_status "Downloading and installing Cursor IDE..."
        # Download Cursor AppImage
        wget -q --show-progress -O cursor.AppImage https://downloader.cursor.sh/linux/appImage/x64
        chmod +x cursor.AppImage
        
        # Create desktop entry and move to applications
        sudo mv cursor.AppImage /usr/local/bin/cursor
        
        # Create desktop entry
        cat > ~/.local/share/applications/cursor.desktop << 'EOF'
[Desktop Entry]
Name=Cursor
Comment=The AI-first code editor
Exec=/usr/local/bin/cursor %U
Icon=cursor
Terminal=false
Type=Application
Categories=Development;TextEditor;
MimeType=text/plain;text/x-chdr;text/x-csrc;text/x-c++hdr;text/x-c++src;text/x-java;text/x-dsrc;text/x-pascal;text/x-perl;text/x-python;application/x-php;application/x-httpd-php3;application/x-httpd-php4;application/x-httpd-php5;application/javascript;application/json;text/css;text/html;text/xml;text/x-sql;text/x-vb;text/x-yaml;
EOF
        
        # Create icon (simple text icon)
        sudo tee /usr/share/pixmaps/cursor.xpm > /dev/null << 'EOF'
/* XPM */
static char * cursor_xpm[] = {
"16 16 2 1",
" 	c None",
".	c #000000",
"                ",
" ............  ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" .          . ",
" ............  ",
"                ",
"                ",
"                "
};
EOF
        
        print_success "Cursor IDE installed successfully"
        print_status "You can now run 'cursor' from the terminal or find it in your applications menu"
    else
        print_success "Cursor IDE is already installed"
    fi
    print_success "Step 8.6 completed: Cursor IDE ready"
    
    # Step 9: Validate installation and run server.py
    print_status "Step 9: Validating installation and starting server..."
    print_warning "Note: This will start the server. Press Ctrl+C to stop after validation."
    
    # Test import of main modules
    python3 -c "
import sys
sys.path.append('.')
sys.path.append('./ServerSideScript')
try:
    import transformers
    import torch
    import librosa
    import numpy as np
    import websockets
    print('✅ All required modules imported successfully')
    print(f'Transformers version: {transformers.__version__}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    print_success "Step 9 completed: Installation validation successful"
    
    # Step 10: Start the server using server.py
    print_status "Step 10: Starting Ultravox server using server.py..."
    print_status "Server will be available at: ws://localhost:8000"
    print_warning "Press Ctrl+C to stop the server when done testing."
    
    # Change to server directory and run server.py
    cd server
    python3 server.py
    
    # Final summary
    print_status "Setup completed successfully!"
    print_status "To start the server:"
    print_status "  1. Activate virtual environment: source venv/bin/activate"
    print_status "  2. Navigate to server directory: cd server"
    print_status "  3. Run server: python server.py"
    print_status "  4. Server will be available at: ws://localhost:8000"
    print_status "  5. Open Cursor IDE: cursor ."
    print_status "Log file saved to: $LOG_FILE"
}

# Run main function
main "$@"