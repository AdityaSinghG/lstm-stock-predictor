import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def train_model(X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, lr=1e-3,
                patience=10, device="cpu"):

    device = torch.device(device)
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_ds)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= len(val_ds)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:4d} | train={t_loss:.6f} | val={v_loss:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[train] Early stop at epoch {epoch} (best val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model, history


def mc_dropout_predict(model, X, n_passes=100, device="cpu", batch_size=256):
    device = torch.device(device)
    model.train()
    X_tensor = torch.tensor(X)
    all_preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            preds = []
            for i in range(0, len(X_tensor), batch_size):
                xb = X_tensor[i:i + batch_size].to(device)
                preds.append(model(xb).cpu().numpy())
            all_preds.append(np.concatenate(preds))

    all_preds = np.stack(all_preds, axis=0)
    return all_preds.mean(axis=0), all_preds.std(axis=0)
