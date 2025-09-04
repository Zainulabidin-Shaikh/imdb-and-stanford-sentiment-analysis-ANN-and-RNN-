"""
Terminal-based training script for sentiment analysis models
Run this script to train and save the best models, then use the UI for predictions
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from data_loader import DataLoader
from ann_model import ANNSentimentClassifier
from rnn_model import RNNSentimentClassifier
from evaluation import ModelEvaluator
import pickle

class ModelTrainer:
    """Terminal-based model trainer"""

    def __init__(self, dataset_name="imdb"):
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name)
        self.evaluator = ModelEvaluator()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories
        self.model_dir = MODELS_DIR / dataset_name
        self.model_dir.mkdir(exist_ok=True)

    def load_and_prepare_data(self):
        """Load and prepare dataset"""
        print(f"Loading {DATASETS[self.dataset_name]['name']} dataset...")

        # Load data
        self.x_train, self.x_test, self.y_train, self.y_test = self.data_loader.load_data()

        # Get data info
        self.data_info = self.data_loader.get_data_info(
            self.x_train, self.y_train, self.x_test, self.y_test
        )

        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.x_train)}")
        print(f"Test samples: {len(self.x_test)}")
        print(f"Vocabulary size: {self.data_info['vocab_size']}")

        # Save tokenizer
        tokenizer_path = self.model_dir / "tokenizer.pkl"
        self.data_loader.save_tokenizer(str(tokenizer_path))
        print(f"Tokenizer saved to: {tokenizer_path}")

        return self.data_info

    def train_ann_model(self, config=None):
        """Train ANN model with optimal configuration"""
        print("\n" + "="*50)
        print("TRAINING ANN MODEL")
        print("="*50)

        if config is None:
            config = {
                'vocab_size': self.data_info['vocab_size'],
                'embedding_dim': 128,  # Increased for better performance
                'max_length': self.data_info['max_sequence_length'],
                'hidden_units': [256, 128, 64],  # Deeper network
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64,  # Larger batch size
                'epochs': 25
            }

        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Create and train model
        ann_model = ANNSentimentClassifier(config)
        model_path = self.model_dir / f"ann_model_{self.timestamp}.h5"

        print(f"\nStarting training...")
        history = ann_model.train(
            self.x_train, self.y_train,
            model_path=str(model_path)
        )

        # Evaluate model
        print("\nEvaluating ANN model...")
        ann_metrics = ann_model.evaluate(self.x_test, self.y_test)

        print("\nANN Model Results:")
        for metric, value in ann_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save model info
        model_info = {
            'model_type': 'ann',
            'dataset': self.dataset_name,
            'config': config,
            'metrics': ann_metrics,
            'timestamp': self.timestamp,
            'model_path': str(model_path),
            'tokenizer_path': str(self.model_dir / "tokenizer.pkl")
        }

        info_path = self.model_dir / f"ann_model_info_{self.timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        return ann_model, ann_metrics, model_info

    def train_rnn_model(self, config=None):
        """Train RNN model with optimal configuration"""
        print("\n" + "="*50)
        print("TRAINING RNN MODEL")
        print("="*50)

        if config is None:
            config = {
                'vocab_size': self.data_info['vocab_size'],
                'embedding_dim': 128,  # Increased for better performance
                'max_length': self.data_info['max_sequence_length'],
                'lstm_units': 128,
                'bidirectional': True,
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64,  # Larger batch size
                'epochs': 25
            }

        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Create and train model
        rnn_model = RNNSentimentClassifier(config)
        model_path = self.model_dir / f"rnn_model_{self.timestamp}.h5"

        print(f"\nStarting training...")
        history = rnn_model.train(
            self.x_train, self.y_train,
            model_path=str(model_path)
        )

        # Evaluate model
        print("\nEvaluating RNN model...")
        rnn_metrics = rnn_model.evaluate(self.x_test, self.y_test)

        print("\nRNN Model Results:")
        for metric, value in rnn_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save model info
        model_info = {
            'model_type': 'rnn',
            'dataset': self.dataset_name,
            'config': config,
            'metrics': rnn_metrics,
            'timestamp': self.timestamp,
            'model_path': str(model_path),
            'tokenizer_path': str(self.model_dir / "tokenizer.pkl")
        }

        info_path = self.model_dir / f"rnn_model_info_{self.timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        return rnn_model, rnn_metrics, model_info

    def compare_and_save_best(self, ann_info, rnn_info):
        """Compare models and save the best one as default"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)

        ann_f1 = ann_info['metrics']['f1_score']
        rnn_f1 = rnn_info['metrics']['f1_score']

        print(f"ANN F1-Score: {ann_f1:.4f}")
        print(f"RNN F1-Score: {rnn_f1:.4f}")

        # Determine best model
        if rnn_f1 > ann_f1:
            best_model = 'rnn'
            best_info = rnn_info
            print(f"\n🏆 RNN model is better by {rnn_f1 - ann_f1:.4f}")
        else:
            best_model = 'ann'
            best_info = ann_info
            print(f"\n🏆 ANN model is better by {ann_f1 - rnn_f1:.4f}")

        # Save best model info as default
        best_model_info = {
            'best_model_type': best_model,
            'best_model_path': best_info['model_path'],
            'tokenizer_path': best_info['tokenizer_path'],
            'dataset': self.dataset_name,
            'ann_info': ann_info,
            'rnn_info': rnn_info,
            'comparison_timestamp': self.timestamp
        }

        # Save to default location for UI to load
        default_path = self.model_dir / "best_models.json"
        with open(default_path, 'w') as f:
            json.dump(best_model_info, f, indent=2)

        print(f"\nBest model info saved to: {default_path}")

        return best_model, best_info

    def train_all(self):
        """Train both models and save the best"""
        print("🚀 Starting complete training pipeline...")
        print(f"Dataset: {DATASETS[self.dataset_name]['name']}")
        print(f"Timestamp: {self.timestamp}")

        # Load data
        data_info = self.load_and_prepare_data()

        # Train both models
        ann_model, ann_metrics, ann_info = self.train_ann_model()
        rnn_model, rnn_metrics, rnn_info = self.train_rnn_model()

        # Compare and save best
        best_model, best_info = self.compare_and_save_best(ann_info, rnn_info)

        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {best_model.upper()}")
        print(f"F1-Score: {best_info['metrics']['f1_score']:.4f}")
        print(f"Accuracy: {best_info['metrics']['accuracy']:.4f}")
        print("\nYou can now run the UI to test the trained models!")
        print("Command: streamlit run app.py")

def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--dataset', choices=['imdb', 'stanford_sentiment'], 
                       default='imdb', help='Dataset to use for training')
    parser.add_argument('--model', choices=['ann', 'rnn', 'both'], 
                       default='both', help='Model to train')

    args = parser.parse_args()

    # Create trainer
    trainer = ModelTrainer(args.dataset)

    if args.model == 'both':
        trainer.train_all()
    elif args.model == 'ann':
        trainer.load_and_prepare_data()
        trainer.train_ann_model()
    elif args.model == 'rnn':
        trainer.load_and_prepare_data()
        trainer.train_rnn_model()

if __name__ == "__main__":
    main()
