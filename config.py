"""
Configuration settings for the Sentiment Analysis Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    "imdb": {
        "name": "IMDB Movie Reviews",
        "description": "Binary sentiment classification of movie reviews",
        "classes": ["Negative", "Positive"],
        "source": "tensorflow.keras.datasets"
    },
    "stanford_sentiment": {
        "name": "Stanford Sentiment Treebank",
        "description": "Fine-grained sentiment analysis dataset",
        "classes": ["Negative", "Positive"], # We'll use binary version
        "source": "huggingface"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "ann": {
        "name": "Artificial Neural Network",
        "type": "dense",
        "default_params": {
            "vocab_size": 10000,
            "embedding_dim": 100,
            "max_length": 200,
            "hidden_units": [128, 64],
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 20
        }
    },
    "rnn": {
        "name": "Recurrent Neural Network (LSTM)",
        "type": "recurrent",
        "default_params": {
            "vocab_size": 10000,
            "embedding_dim": 100,
            "max_length": 200,
            "lstm_units": 128,
            "dropout_rate": 0.5,
            "recurrent_dropout": 0.2,
            "bidirectional": True,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 20
        }
    }
}

# Training configurations
TRAINING_CONFIG = {
    "validation_split": 0.2,
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "min_lr": 1e-7,
    "save_best_only": True,
    "verbose": 1
}

# Visualization configurations
VIZ_CONFIG = {
    "figsize": (10, 6),
    "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
    "background_color": "#F8F9FA",
    "grid_alpha": 0.3
}

# Streamlit app configurations
STREAMLIT_CONFIG = {
    "page_title": "Sentiment Analysis Deep Learning",
    "page_icon": "🎭",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Text preprocessing configurations
TEXT_CONFIG = {
    "max_vocab_size": 10000,
    "max_sequence_length": 200,
    "padding": "post",
    "truncating": "post",
    "oov_token": "<OOV>"
}
