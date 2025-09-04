#!/usr/bin/env python3
"""
Python setup script for Sentiment Analysis Project
Cross-platform setup and training
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is sufficient"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"Current version: {version.major}.{version.minor}")
        return False

    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    return run_command("pip install -r requirements.txt", "Installing requirements")

def models_exist():
    """Check if trained models exist"""
    model_file = Path("saved_models/imdb/best_models.json")
    return model_file.exists()

def train_models(dataset="imdb"):
    """Train the models"""
    return run_command(f"python train_models.py --dataset {dataset}", 
                      f"Training models on {dataset} dataset")

def start_app():
    """Start the Streamlit app"""
    print("🚀 Starting web application...")
    print("Open your browser and go to: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped")

def main():
    print("🎭 Sentiment Analysis Deep Learning Project Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return

    # Install requirements
    if not install_requirements():
        return

    # Check if models exist
    if models_exist():
        print("✅ Trained models found!")
        start_app()
    else:
        print("⚠️  No trained models found")
        print("Training is required (takes about 15-25 minutes)")

        # Ask for dataset choice
        print("\nWhich dataset would you like to train on?")
        print("1) IMDB Movie Reviews (recommended)")
        print("2) Stanford Sentiment Treebank")

        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                dataset = "imdb"
                break
            elif choice == "2":
                dataset = "stanford_sentiment"
                break
            else:
                print("Please enter 1 or 2")

        # Train models
        if train_models(dataset):
            print("\n🎉 Training completed successfully!")
            start_app()
        else:
            print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
