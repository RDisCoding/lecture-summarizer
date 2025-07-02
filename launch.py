#!/usr/bin/env python3
"""
Lecture Transcript Analyzer Launcher
Simple script to check dependencies and launch the Streamlit app
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'whisper',
        'requests',
        'moviepy',
        'pydub',
        'torch'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing

def install_dependencies():
    """Install missing dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        return True
    except subprocess.CalledProcessError:
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'lecture_analyzer_app.py',
        'transcript_generator.py',
        'requirements.txt'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    return missing_files

def main():
    print("🎧 Lecture Transcript Analyzer Launcher")
    print("=" * 50)

    # Check if required files exist
    missing_files = check_files()
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the same directory.")
        return 1

    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"📦 Missing dependencies: {', '.join(missing_deps)}")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            if install_dependencies():
                print("✅ Dependencies installed successfully!")
            else:
                print("❌ Failed to install dependencies. Please run:")
                print("pip install -r requirements.txt")
                return 1
        else:
            print("Please install dependencies manually:")
            print("pip install -r requirements.txt")
            return 1

    # Check for API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️  No PERPLEXITY_API_KEY environment variable found.")
        print("You can either:")
        print("1. Set it as environment variable: export PERPLEXITY_API_KEY='your_key'")
        print("2. Enter it in the app when prompted")
        print()

    # Launch the app
    print("🚀 Launching Streamlit app...")
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'lecture_analyzer_app.py'])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())