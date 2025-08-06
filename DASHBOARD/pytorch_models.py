import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TabularDataset(Dataset):
    def __init__(self, X, y, is_classification=False):
        self.n = X.shape[0]
        self.X = torch.tensor(X, dtype=torch.float32)
        if is_classification: self.y = torch.tensor(y, dtype=torch.long)
        else: self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    def __len__(self): return self.n
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.network(x)

class ClassificationNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)

def train_model(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.squeeze() if len(y_batch.shape) > 1 and y_batch.shape[1] == 1 else y_batch)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

# pytorch_models.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TabularDataset(Dataset):
    def __init__(self, X, y, is_classification=False):
        self.n = X.shape[0]
        self.X = torch.tensor(X, dtype=torch.float32)
        if is_classification:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

class ClassificationNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

def train_model(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.squeeze() if len(y_batch.shape) > 1 and y_batch.shape[1] == 1 else y_batch)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def evaluate_regression(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return mse, r2

def evaluate_classification(model, data_loader, device, class_names, model_name="Rete Neurale"):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch)
            y_pred_labels = torch.argmax(y_pred_logits, dim=1)
            all_preds.append(y_pred_labels.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    # --- CODICE PER LA MATRICE DI CONFUSIONE SPOSTATO QUI ---
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title(f'Matrice di Confusione - {model_name}', fontsize=18, pad=20)
    plt.ylabel('Classe Reale', fontsize=14)
    plt.xlabel('Classe Prevista', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()
    # --- FINE CODICE SPOSTATO ---
    
    return acc, report