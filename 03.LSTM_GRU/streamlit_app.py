#!/usr/bin/env python3
"""
Next Word Prediction Streamlit App

A production-grade web interface for next-word prediction using LSTM and GRU models.

Author: AI Assistant
Date: 2025
"""

import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Tuple, Optional
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .prediction-box {
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .model-comparison {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class NextWordPredictor:
    """
    Production-grade next-word prediction inference engine.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.lstm_model: Optional[tf.keras.Model] = None
        self.gru_model: Optional[tf.keras.Model] = None
        self.tokenizer = None
        self.sequence_length: int = 0
        self.vocab_size: int = 0
        self.models_loaded: bool = False

    @st.cache_resource
    def load_models(_self) -> bool:
        """
        Load trained models and tokenizer with caching.

        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            # Check if model files exist
            lstm_path = "models/lstm_next_word_best.h5"
            gru_path = "models/gru_next_word_best.h5"
            tokenizer_path = "models/tokenizer.pkl"

            missing_files = []
            for path, name in [(lstm_path, "LSTM model"),
                               (gru_path, "GRU model"),
                               (tokenizer_path, "Tokenizer")]:
                if not os.path.exists(path):
                    missing_files.append(f"{name} ({path})")

            if missing_files:
                st.error(f"Missing required files: {', '.join(missing_files)}")
                st.info("Please run the Jupyter notebook first to train the models.")
                return False

            # Load models
            _self.lstm_model = load_model(lstm_path)
            _self.gru_model = load_model(gru_path)

            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                data = pickle.load(f)
                _self.tokenizer = data['tokenizer']
                _self.sequence_length = data['sequence_length']
                _self.vocab_size = data['vocab_size']

            _self.models_loaded = True
            return True

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_input(self, input_text: str) -> np.ndarray:
        """
        Preprocess input text for prediction.

        Args:
            input_text: Input text string

        Returns:
            Processed sequence array
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")

        # Clean and tokenize input
        input_text = input_text.lower().strip()
        if not input_text:
            raise ValueError("Input text cannot be empty.")

        sequence = self.tokenizer.texts_to_sequences([input_text])[0]

        if not sequence:
            raise ValueError("Input text contains no known words.")

        # Pad or truncate to required length
        if len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        else:
            sequence = [0] * (self.sequence_length - len(sequence)) + sequence

        return np.array([sequence])

    def predict_next_word(self, input_text: str, model_type: str = "lstm",
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next word(s) given input text.

        Args:
            input_text: Input text string
            model_type: Type of model to use ("lstm" or "gru")
            top_k: Number of top predictions to return

        Returns:
            List of (word, probability) tuples
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded.")

        # Select model
        model = self.lstm_model if model_type.lower() == "lstm" else self.gru_model

        # Preprocess input
        input_sequence = self.preprocess_input(input_text)

        # Get predictions
        predictions = model.predict(input_sequence, verbose=0)[0]

        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        # Convert indices to words
        word_index = self.tokenizer.word_index
        index_word = {v: k for k, v in word_index.items()}

        results = []
        for idx in top_indices:
            if idx in index_word and idx != 0:  # Skip padding token
                word = index_word[idx]
                probability = predictions[idx]
                results.append((word, float(probability)))

        return results

    def predict_both_models(self, input_text: str, top_k: int = 3) -> dict:
        """
        Predict next word using both LSTM and GRU models.

        Args:
            input_text: Input text string
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions from both models
        """
        return {
            "lstm": self.predict_next_word(input_text, "lstm", top_k),
            "gru": self.predict_next_word(input_text, "gru", top_k)
        }


def display_predictions(predictions: List[Tuple[str, float]], model_name: str) -> None:
    """
    Display prediction results in a formatted way.

    Args:
        predictions: List of (word, probability) tuples
        model_name: Name of the model
    """
    st.markdown(f"### {model_name} Predictions")

    if not predictions:
        st.warning(f"No predictions available from {model_name}")
        return

    # Create a DataFrame for better display
    df = pd.DataFrame(predictions, columns=["Word", "Probability"])
    df["Rank"] = range(1, len(df) + 1)
    df = df[["Rank", "Word", "Probability"]]
    df["Probability"] = df["Probability"].round(4)

    # Display as table
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Create probability chart
    fig = px.bar(
        df,
        x="Word",
        y="Probability",
        title=f"{model_name} - Word Probabilities",
        color="Probability",
        color_continuous_scale="viridis"
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)


def display_model_comparison(predictions_dict: dict) -> None:
    """
    Display side-by-side comparison of model predictions.

    Args:
        predictions_dict: Dictionary with LSTM and GRU predictions
    """
    st.markdown("## 🔄 Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 LSTM Model")
        lstm_preds = predictions_dict.get("lstm", [])
        if lstm_preds:
            for i, (word, prob) in enumerate(lstm_preds, 1):
                st.markdown(f"**{i}.** {word} `({prob:.3f})`")
        else:
            st.warning("No LSTM predictions available")

    with col2:
        st.markdown("### ⚡ GRU Model")
        gru_preds = predictions_dict.get("gru", [])
        if gru_preds:
            for i, (word, prob) in enumerate(gru_preds, 1):
                st.markdown(f"**{i}.** {word} `({prob:.3f})`")
        else:
            st.warning("No GRU predictions available")

    # Highlight agreements and differences
    if lstm_preds and gru_preds:
        lstm_words = [word for word, _ in lstm_preds]
        gru_words = [word for word, _ in gru_preds]

        common_words = set(lstm_words) & set(gru_words)
        different_words = (set(lstm_words) | set(gru_words)) - common_words

        if common_words:
            st.success(f"✅ **Agreement**: Both models predict: {', '.join(common_words)}")

        if different_words:
            st.info(f"ℹ️ **Differences**: Unique predictions: {', '.join(different_words)}")


def display_model_info() -> None:
    """Display information about the models."""
    st.markdown("## 📊 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🧠 LSTM (Long Short-Term Memory)
        - **Architecture**: 2-layer LSTM with dropout
        - **Parameters**: ~2.1M parameters  
        - **Strengths**: Better for complex, long sequences
        - **Use Case**: When accuracy is paramount
        - **Speed**: Slower inference
        """)

    with col2:
        st.markdown("""
        ### ⚡ GRU (Gated Recurrent Unit)
        - **Architecture**: 2-layer GRU with dropout
        - **Parameters**: ~1.6M parameters
        - **Strengths**: Faster training/inference
        - **Use Case**: When speed matters
        - **Speed**: Faster inference
        """)


def main():
    """Main Streamlit application."""
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = NextWordPredictor()

    predictor = st.session_state.predictor

    # Header
    st.markdown('<h1 class="main-header">🧠 Next Word Prediction</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Powered by LSTM and GRU Neural Networks</p>',
        unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        models_loaded = predictor.load_models()

    if not models_loaded:
        st.stop()

    st.success("✅ Models loaded successfully!")

    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Both Models", "LSTM Only", "GRU Only"],
        index=0,
        help="Choose which model(s) to use for prediction"
    )

    # Number of predictions
    top_k = st.sidebar.slider(
        "Number of Predictions",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top word predictions to show"
    )

    # Show model info
    if st.sidebar.checkbox("Show Model Information", value=False):
        display_model_info()

    # Main interface
    st.markdown("## 💭 Enter Text for Next Word Prediction")

    # Input text area
    input_text = st.text_area(
        "Input Text",
        placeholder="Type your text here... (e.g., 'the quick brown fox')",
        height=100,
        help="Enter a phrase or sentence. The model will predict the next word."
    )

    # Example buttons
    st.markdown("**Quick Examples:**")
    example_col1, example_col2, example_col3, example_col4 = st.columns(4)

    with example_col1:
        if st.button("🦊 'the quick brown fox'"):
            input_text = "the quick brown fox"

    with example_col2:
        if st.button("📚 'once upon a time'"):
            input_text = "once upon a time"

    with example_col3:
        if st.button("🌧️ 'it was a dark and stormy'"):
            input_text = "it was a dark and stormy"

    with example_col4:
        if st.button("🤔 'to be or not to'"):
            input_text = "to be or not to"

    # Update text area with example
    if input_text and input_text != st.session_state.get('last_input', ''):
        st.session_state.last_input = input_text
        st.rerun()

    # Prediction button
    if st.button("🔮 Predict Next Word", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("⚠️ Please enter some text first!")
        else:
            try:
                with st.spinner("🤖 Generating predictions..."):
                    start_time = time.time()

                    if selected_model == "Both Models":
                        predictions = predictor.predict_both_models(input_text, top_k)

                        # Display comparison
                        display_model_comparison(predictions)

                        # Individual model results
                        col1, col2 = st.columns(2)
                        with col1:
                            display_predictions(predictions["lstm"], "LSTM")
                        with col2:
                            display_predictions(predictions["gru"], "GRU")

                    elif selected_model == "LSTM Only":
                        predictions = predictor.predict_next_word(input_text, "lstm", top_k)
                        display_predictions(predictions, "LSTM")

                    else:  # GRU Only
                        predictions = predictor.predict_next_word(input_text, "gru", top_k)
                        display_predictions(predictions, "GRU")

                    # Show inference time
                    inference_time = time.time() - start_time
                    st.info(f"⏱️ Prediction completed in {inference_time:.3f} seconds")

            except ValueError as e:
                st.error(f"❌ Input Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please check that the models are properly trained and saved.")

    # Footer with additional information
    st.markdown("---")
    st.markdown("## 📋 How to Use")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1️⃣ Enter Text
        - Type a partial sentence or phrase
        - Use proper grammar and spelling
        - Minimum 2-3 words recommended
        """)

    with col2:
        st.markdown("""
        ### 2️⃣ Choose Model
        - **Both Models**: Compare predictions
        - **LSTM Only**: More complex patterns
        - **GRU Only**: Faster predictions
        """)

    with col3:
        st.markdown("""
        ### 3️⃣ Get Predictions
        - See top predicted words
        - View probability scores
        - Compare model performance
        """)

    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Model Architecture
        - **LSTM**: 2-layer LSTM with 256/128 units + dropout
        - **GRU**: 2-layer GRU with 256/128 units + dropout
        - **Embedding**: 128-dimensional word embeddings
        - **Vocabulary**: ~15,000 most common words
        - **Sequence Length**: 40 words context window

        ### Training Data
        - Multiple classic literature books from Project Gutenberg
        - Preprocessed and tokenized text data
        - Train/validation split for model evaluation

        ### Performance
        - Models trained with early stopping and learning rate reduction
        - Best model checkpoints saved during training
        - Real-time inference with probability scores
        """)


if __name__ == "__main__":
    main()
