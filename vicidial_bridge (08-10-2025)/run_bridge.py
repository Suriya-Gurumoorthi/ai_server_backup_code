#!/usr/bin/env python3
"""
Launcher script for the modular Vicidial Bridge

This script provides an easy way to run the modular voice bot system.
It can be executed directly or used as a systemd service.
"""

import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function
from main import run_bridge

if __name__ == "__main__":
    print("Starting Vicidial Bridge (Modular Version)...")
    print("=" * 50)
    run_bridge()
