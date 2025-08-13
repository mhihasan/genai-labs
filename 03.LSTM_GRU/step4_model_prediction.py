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


class NextWordInference:
    """
    Handles inference for next-word prediction.
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.sequence_length = None
        self.vocab_size = None

    def load_model_and_tokenizer(self) -> None:
        """Load trained model and tokenizer."""
        try:
            # Load model
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")

            # Load tokenizer
            with open(self.tokenizer_path, 'rb') as f:
                data = pickle.load(f)
                self.tokenizer = data['tokenizer']
                self.sequence_length = data['sequence_length']
                self.vocab_size = data['vocab_size']

            print(f"Tokenizer loaded from {self.tokenizer_path}")

        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            raise

    def preprocess_input(self, input_text: str) -> np.ndarray:
        """
        Preprocess input text for prediction.

        Args:
            input_text: Input text string

        Returns:
            Processed sequence array
        """
        # Clean and tokenize input
        input_text = input_text.lower().strip()
        sequence = self.tokenizer.texts_to_sequences([input_text])[0]

        # Pad or truncate to required length
        if len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        else:
            sequence = [0] * (self.sequence_length - len(sequence)) + sequence

        return np.array([sequence])

    def predict_next_word(self, input_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next word(s) given input text.

        Args:
            input_text: Input text string
            top_k: Number of top predictions to return

        Returns:
            List of (word, probability) tuples
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        # Preprocess input
        input_sequence = self.preprocess_input(input_text)

        # Get predictions
        predictions = self.model.predict(input_sequence, verbose=0)[0]

        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        # Convert indices to words
        word_index = self.tokenizer.word_index
        index_word = {v: k for k, v in word_index.items()}

        results = []
        for idx in top_indices:
            if idx in index_word:
                word = index_word[idx]
                probability = predictions[idx]
                results.append((word, float(probability)))

        return results

    def predict_single_word(self, input_text: str) -> str:
        """
        Predict single next word.

        Args:
            input_text: Input text string

        Returns:
            Predicted word
        """
        predictions = self.predict_next_word(input_text, top_k=1)
        return predictions[0][0] if predictions else "<unknown>"


# Test the inference functions
print("\n" + "=" * 50)
print("TESTING INFERENCE")
print("=" * 50)
# Test sentences
test_sentences = [
    "the quick brown fox",
    "once upon a time",
    "it was a dark and stormy",
    "to be or not to",
    "i think therefore i"
]


def predict_with_lstm():
    # Test with LSTM model
    lstm_inference = NextWordInference(
        model_path="models/lstm_next_word_best.h5",
        tokenizer_path="models/tokenizer.pkl"
    )

    try:
        lstm_inference.load_model_and_tokenizer()


        print("LSTM Predictions:")
        print("-" * 40)
        for sentence in test_sentences:
            predictions = lstm_inference.predict_next_word(sentence, top_k=3)
            print(f"Input: '{sentence}'")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word} ({prob:.3f})")
            print()

    except Exception as e:
        print(f"Error in LSTM inference: {e}")


def predict_with_gru():
    # Test with GRU model
    gru_inference = NextWordInference(
        model_path="models/gru_next_word_best.h5",
        tokenizer_path="models/tokenizer.pkl"
    )

    try:
        gru_inference.load_model_and_tokenizer()

        print("GRU Predictions:")
        print("-" * 40)
        for sentence in test_sentences:
            predictions = gru_inference.predict_next_word(sentence, top_k=3)
            print(f"Input: '{sentence}'")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word} ({prob:.3f})")
            print()

    except Exception as e:
        print(f"Error in GRU inference: {e}")




