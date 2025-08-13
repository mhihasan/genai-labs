import os
import re
import pickle
import numpy as np

import tensorflow as tf
import pandas as pd
import requests
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .step2_data_preprocessing import TextPreprocessor, X_val, y_train, y_val, X_train


class NextWordPredictor:
    """
    Next-word prediction model using LSTM or GRU.
    """

    def __init__(self, vocab_size: int, sequence_length: int, embedding_dim: int = 128):
        """
        Initialize the model.

        Args:
            vocab_size: Size of vocabulary
            sequence_length: Length of input sequences
            embedding_dim: Dimensionality of embeddings
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = None

    def build_lstm_model(self, lstm_units: int = 256, dropout_rate: float = 0.3) -> None:
        """
        Build LSTM-based model.

        Args:
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='embedding'
            ),
            LSTM(lstm_units, return_sequences=True, name='lstm_1'),
            Dropout(dropout_rate, name='dropout_1'),
            LSTM(lstm_units//2, name='lstm_2'),
            Dropout(dropout_rate, name='dropout_2'),
            Dense(self.vocab_size, activation='softmax', name='output')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("LSTM Model Architecture:")
        self.model.summary()

    def build_gru_model(self, gru_units: int = 256, dropout_rate: float = 0.3) -> None:
        """
        Build GRU-based model.

        Args:
            gru_units: Number of GRU units
            dropout_rate: Dropout rate for regularization
        """
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='embedding'
            ),
            GRU(gru_units, return_sequences=True, name='gru_1'),
            Dropout(dropout_rate, name='dropout_1'),
            GRU(gru_units//2, name='gru_2'),
            Dropout(dropout_rate, name='dropout_2'),
            Dense(self.vocab_size, activation='softmax', name='output')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("GRU Model Architecture:")
        self.model.summary()

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_name: str, epochs: int = 50, batch_size: int = 128) -> None:
        """
        Train the model with callbacks.

        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            model_name: Name for saving model checkpoints
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Create models directory
        os.makedirs("models", exist_ok=True)

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=f"models/{model_name}_best.h5",
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print(f"Training {model_name} model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        self.model.save(f"models/{model_name}_final.h5")
        print(f"Model saved as models/{model_name}_final.h5")

    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# Train LSTM Model
print("\n" + "="*50)
print("TRAINING LSTM MODEL")
print("="*50)
preprocessor = TextPreprocessor(max_vocab_size=15000, sequence_length=40)

def train_lstm_model():
    lstm_predictor = NextWordPredictor(
        vocab_size=preprocessor.vocab_size,
        sequence_length=preprocessor.sequence_length,
        embedding_dim=128
    )

    lstm_predictor.build_lstm_model(lstm_units=256, dropout_rate=0.3)
    lstm_predictor.train(X_train, y_train, X_val, y_val,
                         model_name="lstm_next_word", epochs=30, batch_size=64)

    # Plot LSTM training history
    lstm_predictor.plot_training_history()


def train_gru():
    # Train GRU Model
    print("\n" + "=" * 50)
    print("TRAINING GRU MODEL")
    print("=" * 50)

    gru_predictor = NextWordPredictor(
        vocab_size=preprocessor.vocab_size,
        sequence_length=preprocessor.sequence_length,
        embedding_dim=128
    )

    gru_predictor.build_gru_model(gru_units=256, dropout_rate=0.3)
    gru_predictor.train(X_train, y_train, X_val, y_val,
                        model_name="gru_next_word", epochs=30, batch_size=64)

    # Plot GRU training history
    gru_predictor.plot_training_history()


if __name__ == "__main__":
    # Train LSTM model
    train_lstm_model()

    # # Train GRU model
    # train_gru()
    #
    # print("\n" + "="*50)
    # print("TRAINING COMPLETED")
    # print("="*50)
