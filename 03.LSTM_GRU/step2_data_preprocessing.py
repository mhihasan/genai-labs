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
from .data_collection import DataCollector

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

class TextPreprocessor:
    """
    Handles text preprocessing for next-word prediction models.
    """

    def __init__(self, max_vocab_size: int = 10000, sequence_length: int = 50):
        """
        Initialize the text preprocessor.

        Args:
            max_vocab_size: Maximum vocabulary size
            sequence_length: Length of input sequences
        """
        self.max_vocab_size = max_vocab_size
        self.sequence_length = sequence_length
        self.tokenizer = None
        self.vocab_size = 0

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text: Raw text string

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove extra whitespace
        text = text.strip()

        return text

    def create_tokenizer(self, text: str) -> None:
        """
        Create and fit tokenizer on text data.

        Args:
            text: Cleaned text data
        """
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token="<OOV>",
            filters='',  # We already cleaned the text
        )

        # Split into sentences for better tokenization
        sentences = text.split('.')
        self.tokenizer.fit_on_texts(sentences)

        # Update vocab size (add 1 for OOV token)
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_vocab_size)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {list(self.tokenizer.word_index.keys())[:20]}")

    def create_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and targets for training.

        Args:
            text: Cleaned text data

        Returns:
            Tuple of (input_sequences, targets)
        """
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences([text])[0]

        # Create input-target pairs
        input_sequences = []
        targets = []

        print("Creating training sequences...")
        for i in tqdm(range(self.sequence_length, len(sequences))):
            input_seq = sequences[i-self.sequence_length:i]
            target = sequences[i]

            input_sequences.append(input_seq)
            targets.append(target)

        # Convert to numpy arrays
        X = np.array(input_sequences)
        y = np.array(targets)

        print(f"Created {len(X):,} training sequences")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        return X, y

    def save_tokenizer(self, filepath: str) -> None:
        """Save tokenizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length,
                'max_vocab_size': self.max_vocab_size
            }, f)
        print(f"Tokenizer saved to {filepath}")

    def load_tokenizer(self, filepath: str) -> None:
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.tokenizer = data['tokenizer']
            self.vocab_size = data['vocab_size']
            self.sequence_length = data['sequence_length']
            self.max_vocab_size = data['max_vocab_size']
        print(f"Tokenizer loaded from {filepath}")

# Initialize preprocessor
preprocessor = TextPreprocessor(max_vocab_size=15000, sequence_length=40)

collector = DataCollector()

# Load and preprocess text data
raw_text = collector.load_text_data("combined_books.txt")
if not raw_text:
    print("No data found. Please run the data collection section first.")
else:
    # Clean text
    print("Cleaning text...")
    clean_text = preprocessor.clean_text(raw_text)

    # Create tokenizer
    print("Creating tokenizer...")
    preprocessor.create_tokenizer(clean_text)

    # Create training sequences
    print("Creating training sequences...")
    X, y = preprocessor.create_sequences(clean_text)

    # Save tokenizer
    os.makedirs("models", exist_ok=True)
    preprocessor.save_tokenizer("models/tokenizer.pkl")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Training targets shape: {y_train.shape}")
print(f"Validation targets shape: {y_val.shape}")

