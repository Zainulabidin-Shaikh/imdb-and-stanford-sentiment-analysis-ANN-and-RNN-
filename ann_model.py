"""
Artificial Neural Network model for sentiment classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout, GlobalAveragePooling1D,
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from typing import Dict, Any, Tuple
import numpy as np
from config import MODEL_CONFIGS, TRAINING_CONFIG

class ANNSentimentClassifier:
    """Artificial Neural Network for sentiment classification"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or MODEL_CONFIGS["ann"]["default_params"]
        self.model = None
        self.history = None

    def build_model(self) -> tf.keras.Model:
        """Build the ANN model architecture"""

        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.config["vocab_size"],
                output_dim=self.config["embedding_dim"],
                input_length=self.config["max_length"],
                name="embedding"
            ),

            # Global average pooling to convert sequences to fixed-size vectors
            GlobalAveragePooling1D(name="global_avg_pooling"),

            # Batch normalization
            BatchNormalization(name="batch_norm_1"),

            # First hidden layer
            Dense(
                self.config["hidden_units"][0],
                kernel_regularizer=l2(0.001),
                name="dense_1"
            ),
            Activation("relu", name="activation_1"),
            Dropout(self.config["dropout_rate"], name="dropout_1"),

            # Second hidden layer
            Dense(
                self.config["hidden_units"][1],
                kernel_regularizer=l2(0.001),
                name="dense_2"
            ),
            Activation("relu", name="activation_2"),
            Dropout(self.config["dropout_rate"], name="dropout_2"),

            # Batch normalization
            BatchNormalization(name="batch_norm_2"),

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
              model_path: str = "ann_model.h5") -> tf.keras.callbacks.History:
        """Train the ANN model"""

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
        print("Training ANN model...")
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

def create_ann_model(vocab_size: int, embedding_dim: int, max_length: int,
                    hidden_units: list = None, dropout_rate: float = 0.5,
                    learning_rate: float = 0.001) -> ANNSentimentClassifier:
    """Create ANN model with custom parameters"""

    if hidden_units is None:
        hidden_units = [128, 64]

    config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "hidden_units": hidden_units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": 32,
        "epochs": 20
    }

    return ANNSentimentClassifier(config)
