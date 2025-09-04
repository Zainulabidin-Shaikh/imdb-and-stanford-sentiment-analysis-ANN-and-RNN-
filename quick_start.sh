#!/bin/bash

# 🎭 Sentiment Analysis Project - Quick Start Script
echo "🎭 Sentiment Analysis Deep Learning Project"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"

if ! python -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" ]]; then
    # Windows Git Bash
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Check if models exist
if [ -f "saved_models/imdb/best_models.json" ]; then
    echo -e "${GREEN}✅ Models already trained!${NC}"
    echo -e "${BLUE}Starting web application...${NC}"
    streamlit run app.py
else
    echo -e "${YELLOW}⚠️  No trained models found${NC}"
    echo -e "${BLUE}Starting model training...${NC}"
    echo "This will take about 15-25 minutes."
    echo ""

    # Ask user for dataset choice
    echo "Which dataset would you like to train on?"
    echo "1) IMDB Movie Reviews (recommended)"
    echo "2) Stanford Sentiment Treebank"
    read -p "Enter choice (1 or 2): " choice

    case $choice in
        1)
            dataset="imdb"
            ;;
        2)
            dataset="stanford_sentiment"
            ;;
        *)
            echo "Invalid choice. Using IMDB dataset."
            dataset="imdb"
            ;;
    esac

    # Train models
    echo -e "${BLUE}Training models on $dataset dataset...${NC}"
    python train_models.py --dataset $dataset

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 Training completed successfully!${NC}"
        echo -e "${BLUE}Starting web application...${NC}"
        streamlit run app.py
    else
        echo -e "${RED}❌ Training failed. Check the error messages above.${NC}"
        exit 1
    fi
fi

# Deactivate virtual environment on exit
trap 'deactivate' EXIT
