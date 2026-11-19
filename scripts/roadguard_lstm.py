
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT = Path("experiments/lstm")
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"

SEQ_LEN = 6
NUM_SEQUENCES = 600

def generate_data():
    sequences = []
    targets = []

    for _ in range(NUM_SEQUENCES):
        start = np.random.uniform(0.05, 0.5)
        growth = np.random.uniform(0.005, 0.04)

        values = [
            min(1.0, start + growth * t)
            for t in range(SEQ_LEN + 1)
        ]

        sequences.append(values[:-1])
        targets.append(values[-1])

    x = torch.tensor(
        np.array(sequences, dtype=np.float32)
    ).unsqueeze(-1)

    y = torch.tensor(
        np.array(targets, dtype=np.float32)
    ).unsqueeze(-1)

    return x, y

class RoadDeteriorationLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

def main():
    x, y = generate_data()

    # Fixed split for reproducibility.
    indices = torch.randperm(len(x))

    train_idx = indices[:480]
    test_idx = indices[480:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    model = RoadDeteriorationLSTM().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005
    )

    criterion = nn.MSELoss()

    print("===== ROADGUARD LSTM TRAINING =====")

    for epoch in range(30):
        model.train()

        optimizer.zero_grad()

        predictions = model(x_train)
        loss = criterion(predictions, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/30 "
                f"Loss: {loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        predictions = model(x_test)

        mae = torch.mean(
            torch.abs(predictions - y_test)
        ).item()

        rmse = torch.sqrt(
            torch.mean(
                (predictions - y_test) ** 2
            )
        ).item()

    torch.save(
        model.state_dict(),
        OUTPUT / "roadguard_lstm.pt"
    )

    report = {
        "model": "LSTM",
        "data": "Synthetic road-condition sequences",
        "sequence_length": SEQ_LEN,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "epochs": 30,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "limitations": (
            "Synthetic demonstration only. "
            "Not validated for real-world "
            "road deterioration forecasting."
        )
    }

    with open(
        OUTPUT / "lstm_results.json",
        "w"
    ) as f:
        json.dump(report, f, indent=4)

    print("\n===== LSTM RESULTS =====")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("Model saved successfully")
    print("LSTM EXPERIMENT COMPLETED")

if __name__ == "__main__":
    main()
