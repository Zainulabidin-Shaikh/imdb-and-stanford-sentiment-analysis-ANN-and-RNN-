"""
Data loading and preprocessing utilities for sentiment analysis
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle
from typing import Tuple, Dict, Any
from config import *

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataLoader:
    """Data loader and preprocessor for sentiment analysis datasets"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.tokenizer = None
        self.max_length = TEXT_CONFIG["max_sequence_length"]
        self.vocab_size = TEXT_CONFIG["max_vocab_size"]

    def load_imdb_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess IMDB dataset"""
        print("Loading IMDB dataset...")

        # Load IMDB dataset from Keras
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=self.vocab_size,
            maxlen=self.max_length
        )

        # Convert back to text for preprocessing
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        def decode_review(text):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

        x_train_text = [decode_review(review) for review in x_train]
        x_test_text = [decode_review(review) for review in x_test]

        # Combine for consistent preprocessing
        all_texts = x_train_text + x_test_text
        all_labels = np.concatenate([y_train, y_test])

        return self._preprocess_texts(all_texts, all_labels)

    def load_stanford_sentiment_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess Stanford Sentiment Treebank dataset"""
        print("Loading Stanford Sentiment Treebank dataset...")

        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("sst2")

            # Extract texts and labels
            train_texts = dataset['train']['sentence']
            train_labels = dataset['train']['label']

            validation_texts = dataset['validation']['sentence']
            validation_labels = dataset['validation']['label']

            # Combine train and validation for consistent preprocessing
            all_texts = train_texts + validation_texts
            all_labels = train_labels + validation_labels

            return self._preprocess_texts(all_texts, all_labels)

        except Exception as e:
            print(f"Error loading Stanford Sentiment dataset: {e}")
            print("Falling back to IMDB dataset...")
            return self.load_imdb_data()

    def _preprocess_texts(self, texts: list, labels: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess texts and split into train/test sets"""

        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            cleaned_texts, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )

        # Create and fit tokenizer
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token=TEXT_CONFIG["oov_token"]
        )
        self.tokenizer.fit_on_texts(x_train)

        # Convert texts to sequences
        x_train_seq = self.tokenizer.texts_to_sequences(x_train)
        x_test_seq = self.tokenizer.texts_to_sequences(x_test)

        # Pad sequences
        x_train_padded = pad_sequences(
            x_train_seq,
            maxlen=self.max_length,
            padding=TEXT_CONFIG["padding"],
            truncating=TEXT_CONFIG["truncating"]
        )

        x_test_padded = pad_sequences(
            x_test_seq,
            maxlen=self.max_length,
            padding=TEXT_CONFIG["padding"],
            truncating=TEXT_CONFIG["truncating"]
        )

        return x_train_padded, x_test_padded, np.array(y_train), np.array(y_test)

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data based on dataset name"""
        if self.dataset_name == "imdb":
            return self.load_imdb_data()
        elif self.dataset_name == "stanford_sentiment":
            return self.load_stanford_sentiment_data()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def get_data_info(self, x_train: np.ndarray, y_train: np.ndarray, 
                     x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive data information"""

        # Calculate sequence lengths
        train_lengths = [len([x for x in seq if x != 0]) for seq in x_train]
        test_lengths = [len([x for x in seq if x != 0]) for seq in x_test]

        info = {
            "dataset_name": DATASETS[self.dataset_name]["name"],
            "total_samples": len(x_train) + len(x_test),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "vocab_size": len(self.tokenizer.word_index) + 1,
            "max_sequence_length": self.max_length,
            "class_distribution": {
                "train": {
                    "negative": int(np.sum(y_train == 0)),
                    "positive": int(np.sum(y_train == 1))
                },
                "test": {
                    "negative": int(np.sum(y_test == 0)),
                    "positive": int(np.sum(y_test == 1))
                }
            },
            "sequence_length_stats": {
                "train": {
                    "mean": np.mean(train_lengths),
                    "std": np.std(train_lengths),
                    "min": np.min(train_lengths),
                    "max": np.max(train_lengths),
                    "median": np.median(train_lengths)
                },
                "test": {
                    "mean": np.mean(test_lengths),
                    "std": np.std(test_lengths),
                    "min": np.min(test_lengths),
                    "max": np.max(test_lengths),
                    "median": np.median(test_lengths)
                }
            }
        }

        return info

    def save_tokenizer(self, filepath: str):
        """Save tokenizer for later use"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load_tokenizer(self, filepath: str):
        """Load saved tokenizer"""
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def preprocess_single_text(self, text: str) -> np.ndarray:
        """Preprocess a single text for prediction"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Please load data first.")

        # Clean text
        cleaned_text = self._clean_text(text)

        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        # Pad sequence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding=TEXT_CONFIG["padding"],
            truncating=TEXT_CONFIG["truncating"]
        )

        return padded_sequence

def get_word_frequencies(tokenizer: Tokenizer, top_n: int = 100) -> Dict[str, int]:
    """Get top N word frequencies from tokenizer"""
    word_freq = {}

    for word, index in tokenizer.word_index.items():
        if index <= top_n:
            word_freq[word] = tokenizer.word_counts.get(word, 0)

    return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
