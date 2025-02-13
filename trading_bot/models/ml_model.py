import torch
import numpy as np
import os

MODEL_PATH = "trading_bot/models/trading_model_final.pth"
X_TEST_PATH = "trading_bot/data/X.npy"

class TradingModel(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        super(TradingModel, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.BatchNorm1d(hidden_size // 2),
            torch.nn.LeakyReLU(0.01),

            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.LeakyReLU(0.01),

            torch.nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    """
    Load the trained model from file.
    """
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found!")
        return None

    model = TradingModel(input_size=5, hidden_size=64, output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")
    return model

def predict_trade_signal():
    """
    Make a trade prediction using the trained model.
    """
    if not os.path.exists(X_TEST_PATH):
        print("❌ Processed data file not found!")
        return None

    # Load test data
    X_test = np.load(X_TEST_PATH)
    X_test_tensor = torch.tensor(X_test[-1], dtype=torch.float32).unsqueeze(0)  # Get the latest data point

    # Load model
    model = load_model()
    if model is None:
        return None

    # Make prediction
    with torch.no_grad():
        prediction = model(X_test_tensor).item()

    signal = "BUY" if prediction > 0.5 else "SELL"
    print(f"🔮 Prediction: {prediction:.4f} → Signal: {signal}")
    return signal

if __name__ == "__main__":
    predict_trade_signal()