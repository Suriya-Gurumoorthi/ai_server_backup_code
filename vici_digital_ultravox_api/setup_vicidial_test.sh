#!/bin/bash

# Ultravox Vicidial Integration Setup Script
# This script automates the setup process for testing Ultravox with Vicidial

set -e  # Exit on any error

echo "🚀 Ultravox Vicidial Integration Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python 3.8+ is installed
check_python() {
    print_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    # Install requirements
    if [ -f "requirements_vicidial_test.txt" ]; then
        pip3 install -r requirements_vicidial_test.txt
        print_status "Vicidial test dependencies installed"
    else
        print_warning "requirements_vicidial_test.txt not found"
    fi
    
    if [ -f "requirements_ultravox.txt" ]; then
        pip3 install -r requirements_ultravox.txt
        print_status "Ultravox dependencies installed"
    else
        print_warning "requirements_ultravox.txt not found"
    fi
}

# Create configuration file
create_config() {
    print_info "Creating configuration file..."
    
    if [ ! -f "vicidial_config.json" ]; then
        print_error "vicidial_config.json template not found"
        exit 1
    fi
    
    # Create user config if it doesn't exist
    if [ ! -f "my_vicidial_config.json" ]; then
        cp vicidial_config.json my_vicidial_config.json
        print_status "Configuration template copied to my_vicidial_config.json"
        print_warning "Please edit my_vicidial_config.json with your Vicidial server details"
    else
        print_info "my_vicidial_config.json already exists"
    fi
}

# Check Ultravox installation
check_ultravox() {
    print_info "Checking Ultravox installation..."
    
    if [ -f "check_ultravox_models.py" ]; then
        if python3 check_ultravox_models.py; then
            print_status "Ultravox models are available"
        else
            print_warning "Ultravox models may not be properly loaded"
        fi
    else
        print_warning "check_ultravox_models.py not found - skipping Ultravox check"
    fi
}

# Make scripts executable
make_executable() {
    print_info "Making scripts executable..."
    
    chmod +x ultravox_vicidial_test.py
    chmod +x quick_vicidial_test.py
    print_status "Scripts made executable"
}

# Create test directory
create_test_dir() {
    print_info "Creating test output directory..."
    
    mkdir -p vicidial_test_output
    print_status "Test output directory created"
}

# Display next steps
show_next_steps() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "================================"
    echo ""
    echo "Next steps:"
    echo "1. Edit my_vicidial_config.json with your Vicidial server details:"
    echo "   - server_url: Your Vicidial server URL"
    echo "   - api_key: Your API key"
    echo "   - test_phone_number: Phone number for testing"
    echo ""
    echo "2. Test basic connectivity:"
    echo "   python3 quick_vicidial_test.py"
    echo ""
    echo "3. Run comprehensive tests:"
    echo "   python3 ultravox_vicidial_test.py --config my_vicidial_config.json --test-type comprehensive"
    echo ""
    echo "4. View logs:"
    echo "   tail -f ultravox_vicidial_test.log"
    echo ""
    echo "For detailed instructions, see: VICIDIAL_INTEGRATION_GUIDE.md"
    echo ""
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    check_python
    install_dependencies
    create_config
    check_ultravox
    make_executable
    create_test_dir
    
    show_next_steps
}

# Run main function
main "$@"
