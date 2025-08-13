from keras.src.saving import load_model
from .step2_data_preprocessing import X_val, y_val

def evaluate_models() -> None:
    """Evaluate both models on validation set."""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Evaluate LSTM
    try:
        lstm_model = load_model("models/lstm_next_word_best.h5")
        lstm_loss, lstm_acc = lstm_model.evaluate(X_val, y_val, verbose=0)
        print(f"LSTM - Validation Loss: {lstm_loss:.4f}, Accuracy: {lstm_acc:.4f}")
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")

    # Evaluate GRU
    try:
        gru_model = load_model("models/gru_next_word_best.h5")
        gru_loss, gru_acc = gru_model.evaluate(X_val, y_val, verbose=0)
        print(f"GRU - Validation Loss: {gru_loss:.4f}, Accuracy: {gru_acc:.4f}")
    except Exception as e:
        print(f"Error evaluating GRU: {e}")


evaluate_models()

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print("Files created:")
print("- models/lstm_next_word_best.h5 (Best LSTM model)")
print("- models/gru_next_word_best.h5 (Best GRU model)")
print("- models/tokenizer.pkl (Tokenizer and preprocessing config)")
