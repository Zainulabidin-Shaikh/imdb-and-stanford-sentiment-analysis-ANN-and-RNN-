"""
RNN/LSTM model for sentiment classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    BatchNormalization, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from typing import Dict, Any, Tuple
import numpy as np
from config import MODEL_CONFIGS, TRAINING_CONFIG

class RNNSentimentClassifier:
    """RNN/LSTM model for sentiment classification"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or MODEL_CONFIGS["rnn"]["default_params"]
        self.model = None
        self.history = None

    def build_model(self) -> tf.keras.Model:
        """Build the RNN/LSTM model architecture"""

        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.config["vocab_size"],
                output_dim=self.config["embedding_dim"],
                input_length=self.config["max_length"],
                name="embedding"
            ),

            # Spatial dropout for regularization
            SpatialDropout1D(0.2, name="spatial_dropout"),

            # LSTM layer (bidirectional if specified)
            self._create_lstm_layer(),

            # Batch normalization
            BatchNormalization(name="batch_norm"),

            # Dense layer
            Dense(
                64,
                activation="relu",
                kernel_regularizer=l2(0.001),
                name="dense_1"
            ),

            # Dropout
            Dropout(self.config["dropout_rate"], name="dropout"),

            # Output layer
            Dense(1, activation="sigmoid", name="output")
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )

        self.model = model
        return model

    def _create_lstm_layer(self):
        """Create LSTM layer (bidirectional or unidirectional)"""
        lstm_layer = LSTM(
            self.config["lstm_units"],
            dropout=self.config["dropout_rate"],
            recurrent_dropout=self.config["recurrent_dropout"],
            kernel_regularizer=l2(0.001),
            name="lstm"
        )

        if self.config["bidirectional"]:
            return Bidirectional(lstm_layer, name="bidirectional_lstm")
        else:
            return lstm_layer

    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            self.build_model()

        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        self.model.summary()

        sys.stdout = old_stdout
        summary = buffer.getvalue()

        return summary

    def create_callbacks(self, model_path: str) -> list:
        """Create training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=TRAINING_CONFIG["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=TRAINING_CONFIG["reduce_lr_patience"],
                min_lr=TRAINING_CONFIG["min_lr"],
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=TRAINING_CONFIG["save_best_only"],
                verbose=1
            )
        ]

        return callbacks

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray = None, y_val: np.ndarray = None,
              model_path: str = "rnn_model.h5") -> tf.keras.callbacks.History:
        """Train the RNN model"""

        if self.model is None:
            self.build_model()

        # Prepare validation data
        if x_val is None or y_val is None:
            validation_data = None
            validation_split = TRAINING_CONFIG["validation_split"]
        else:
            validation_data = (x_val, y_val)
            validation_split = None

        # Create callbacks
        callbacks = self.create_callbacks(model_path)

        # Train model
        print("Training RNN/LSTM model...")
        print(f"Training samples: {len(x_train)}")
        if validation_data is not None:
            print(f"Validation samples: {len(x_val)}")

        self.history = self.model.fit(
            x_train, y_train,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=TRAINING_CONFIG["verbose"]
        )

        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        probabilities = self.predict(x)
        # Convert to class probabilities [negative_prob, positive_prob]
        negative_prob = 1 - probabilities.flatten()
        positive_prob = probabilities.flatten()

        return np.column_stack([negative_prob, positive_prob])

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get predictions
        y_pred_proba = self.predict(x_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        return metrics

    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.save(filepath)

    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]):
        """Update model configuration"""
        self.config.update(new_config)
        # Rebuild model if it exists
        if self.model is not None:
            self.build_model()

class AdvancedRNNClassifier(RNNSentimentClassifier):
    """Advanced RNN with multiple LSTM layers and attention"""

    def build_model(self) -> tf.keras.Model:
        """Build advanced RNN model with multiple layers"""

        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.config["vocab_size"],
                output_dim=self.config["embedding_dim"],
                input_length=self.config["max_length"],
                name="embedding"
            ),

            # Spatial dropout
            SpatialDropout1D(0.2, name="spatial_dropout"),

            # First LSTM layer (return sequences for stacking)
            self._create_lstm_layer(return_sequences=True, name_suffix="1"),

            # Second LSTM layer
            self._create_lstm_layer(return_sequences=False, name_suffix="2"),

            # Batch normalization
            BatchNormalization(name="batch_norm"),

            # Dense layers
            Dense(
                128,
                activation="relu",
                kernel_regularizer=l2(0.001),
                name="dense_1"
            ),
            Dropout(self.config["dropout_rate"], name="dropout_1"),

            Dense(
                64,
                activation="relu",
                kernel_regularizer=l2(0.001),
                name="dense_2"
            ),
            Dropout(self.config["dropout_rate"], name="dropout_2"),

            # Output layer
            Dense(1, activation="sigmoid", name="output")
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )

        self.model = model
        return model

    def _create_lstm_layer(self, return_sequences=False, name_suffix=""):
        """Create LSTM layer with custom parameters"""
        lstm_layer = LSTM(
            self.config["lstm_units"],
            dropout=self.config["dropout_rate"],
            recurrent_dropout=self.config["recurrent_dropout"],
            kernel_regularizer=l2(0.001),
            return_sequences=return_sequences,
            name=f"lstm{name_suffix}"
        )

        if self.config["bidirectional"]:
            return Bidirectional(lstm_layer, name=f"bidirectional_lstm{name_suffix}")
        else:
            return lstm_layer

def create_rnn_model(vocab_size: int, embedding_dim: int, max_length: int,
                    lstm_units: int = 128, bidirectional: bool = True,
                    dropout_rate: float = 0.5, learning_rate: float = 0.001) -> RNNSentimentClassifier:
    """Create RNN model with custom parameters"""

    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "lstm_units": lstm_units,
        "bidirectional": bidirectional,
        "dropout_rate": dropout_rate,
        "recurrent_dropout": 0.2,
        "learning_rate": learning_rate,
        "batch_size": 32,
        "epochs": 20
    }

    return RNNSentimentClassifier(config)

def create_advanced_rnn_model(vocab_size: int, embedding_dim: int, max_length: int,
                             lstm_units: int = 128, bidirectional: bool = True,
                             dropout_rate: float = 0.5, learning_rate: float = 0.001) -> AdvancedRNNClassifier:
    """Create advanced RNN model with multiple layers"""

    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "lstm_units": lstm_units,
        "bidirectional": bidirectional,
        "dropout_rate": dropout_rate,
        "recurrent_dropout": 0.2,
        "learning_rate": learning_rate,
        "batch_size": 32,
        "epochs": 20
    }

    return AdvancedRNNClassifier(config)
