"""
Training Script for CTLP Feasibility GNN Models

Runs 4-fold cross-validation over a pre-built PyTorch Geometric dataset (.pt files)
and reports per-fold accuracy, false-positive rate, false-negative rate, and a final
confusion matrix + ROC curve.

Usage
-----
python src/train.py \\
    --feasible-dir  data/processed/feasible/raw_1M/v5/ \\
    --infeasible-dir data/processed/infeasible/raw_1M/v5/ \\
    --model HeatConv_raw_att \\
    --epochs 100 \\
    --batch-size 512 \\
    --lr 0.003 \\
    --folds 4 \\
    --seed 42 \\
    --save-path models/my_model.pth

The --feasible-dir and --infeasible-dir arguments expect directories that contain
pre-processed .pt files (each file is one torch_geometric.data.Data object with labels
already set as data.y = 0 (infeasible) or 1 (feasible)).

To generate the .pt files from raw JSON instances see src/data_preprocessing.py and
src/graph_encoding.py.
"""

import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
from torch_geometric import data as loader

from models import (
    GCN_raw_mean, GCN_raw_att, GCN_n2v_att, GCN_n2v_mean,
    GAT_raw_mean, GAT_raw_att, GAT_n2v_mean,
    GATv2_raw_mean, GATv2_raw_att, GATv2_n2v_mean,
    Transformer_raw_mean, Transformer_raw_mean_v2, Transformer_raw_att, Transformer_raw_att_v2,
    HeatConv_raw_mean, HeatConv_raw_att, HeatConv_raw_mean_v2, HeatConv_raw_att_v2, HeatConv_raw_set_v2,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'GCN_raw_mean': GCN_raw_mean,
    'GCN_raw_att': GCN_raw_att,
    'GCN_n2v_att': GCN_n2v_att,
    'GCN_n2v_mean': GCN_n2v_mean,
    'GAT_raw_mean': GAT_raw_mean,
    'GAT_raw_att': GAT_raw_att,
    'GAT_n2v_mean': GAT_n2v_mean,
    'GATv2_raw_mean': GATv2_raw_mean,
    'GATv2_raw_att': GATv2_raw_att,
    'GATv2_n2v_mean': GATv2_n2v_mean,
    'Transformer_raw_mean': Transformer_raw_mean,
    'Transformer_raw_mean_v2': Transformer_raw_mean_v2,
    'Transformer_raw_att': Transformer_raw_att,
    'Transformer_raw_att_v2': Transformer_raw_att_v2,
    'HeatConv_raw_mean': HeatConv_raw_mean,
    'HeatConv_raw_att': HeatConv_raw_att,
    'HeatConv_raw_mean_v2': HeatConv_raw_mean_v2,
    'HeatConv_raw_att_v2': HeatConv_raw_att_v2,
    'HeatConv_raw_set_v2': HeatConv_raw_set_v2,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(feasible_dir: str, infeasible_dir: str, seed: int = 42) -> list:
    """Load pre-processed .pt graph files from feasible and infeasible directories."""
    feasible_data, infeasible_data = [], []

    for f in os.listdir(feasible_dir):
        if f.endswith('.pt'):
            feasible_data.append(torch.load(os.path.join(feasible_dir, f)))

    for f in os.listdir(infeasible_dir):
        if f.endswith('.pt'):
            infeasible_data.append(torch.load(os.path.join(infeasible_dir, f)))

    dataset = feasible_data + infeasible_data
    random.seed(seed)
    random.shuffle(dataset)
    print(f'Dataset: {len(feasible_data)} feasible + {len(infeasible_data)} infeasible '
          f'= {len(dataset)} total')
    return dataset


def create_data_loaders(train_idx, test_idx, dataset, batch_size: int = 512):
    """Split dataset indices into train/test DataLoaders."""
    train_subset = [dataset[i] for i in train_idx]
    test_subset = [dataset[i] for i in test_idx]
    train_loader = loader.DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    test_loader = loader.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, criterion, data_loader, device):
    """Run a single training epoch; return average loss."""
    model.train()
    total_loss = 0.0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """Evaluate model; return (accuracy, labels, preds, per-sample prediction times)."""
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels, pred_times = [], [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            t0 = time.time()
            out = model(data)
            t1 = time.time()
            pred = out.argmax(dim=1)

            batch_size = data.y.size(0)
            per_sample = (t1 - t0) / batch_size
            pred_times.extend([per_sample] * batch_size)

            correct += (pred == data.y).sum().item()
            total += batch_size
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    return correct / total, all_labels, all_preds, pred_times


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_kfold(model_cls, dataset, device, n_folds=4, epochs=100, batch_size=512,
              lr=0.003, seed=42, save_path=None):
    """Train and evaluate with KFold cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    criterion = torch.nn.CrossEntropyLoss()

    fold_accuracies, fold_train_accuracies = [], []
    all_test_labels, all_test_preds = [], []
    avg_fpr_list, avg_fnr_list = [], []
    all_pred_times = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f'\n=== Fold {fold + 1} / {n_folds} ===')
        train_loader, test_loader = create_data_loaders(train_idx, test_idx, dataset, batch_size)

        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            if epoch % 10 == 0:
                print(f'  Epoch {epoch:3d} | Loss: {loss:.4f}')

        train_acc, _, _, _ = evaluate(model, train_loader, device)
        test_acc, test_labels, test_preds, fold_pred_times = evaluate(model, test_loader, device)

        fold_train_accuracies.append(train_acc)
        fold_accuracies.append(test_acc)
        all_test_labels.extend(test_labels)
        all_test_preds.extend(test_preds)
        all_pred_times.extend(fold_pred_times)

        cm = confusion_matrix(test_labels, test_preds)
        fpr_val = cm[0][1] / (cm[0][1] + cm[0][0])
        fnr_val = cm[1][0] / (cm[1][0] + cm[1][1])
        avg_fpr_list.append(fpr_val)
        avg_fnr_list.append(fnr_val)

        print(f'  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
        print(f'  FPR: {fpr_val:.4f} | FNR: {fnr_val:.4f}')
        print(f'  Avg prediction time: {np.mean(fold_pred_times):.6f}s')
        print(f'  Confusion Matrix:\n{cm}')

        _plot_confusion_matrix(cm, fold + 1)

    # Summary
    print('\n=== Cross-Validation Summary ===')
    print(f'Avg Train Accuracy : {np.mean(fold_train_accuracies):.4f}')
    print(f'Avg Test Accuracy  : {np.mean(fold_accuracies):.4f}')
    print(f'Avg FPR            : {np.mean(avg_fpr_list):.4f}')
    print(f'Avg FNR            : {np.mean(avg_fnr_list):.4f}')
    print(f'Avg Prediction Time: {np.mean(all_pred_times):.6f}s')

    _plot_roc_curve(all_test_labels, all_test_preds)
    _plot_prediction_time_distribution(all_pred_times)

    if save_path:
        torch.save(model, save_path)
        print(f'\nModel saved to {save_path}')

    return model


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm, fold: int):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix — Fold {fold}')
    plt.tight_layout()
    plt.show()


def _plot_roc_curve(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (all folds combined)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def _plot_prediction_time_distribution(pred_times):
    plt.figure()
    sns.boxplot(x=pred_times, color='darkorange')
    plt.xlabel('Prediction time (seconds)')
    plt.title('Distribution of per-sample prediction times')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train a GNN on CTLP feasibility data')
    p.add_argument('--feasible-dir', required=True,
                   help='Directory with pre-processed feasible .pt files')
    p.add_argument('--infeasible-dir', required=True,
                   help='Directory with pre-processed infeasible .pt files')
    p.add_argument('--model', default='HeatConv_raw_att',
                   choices=list(MODEL_REGISTRY.keys()),
                   help='Model architecture to train (default: HeatConv_raw_att)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-path', default=None,
                   help='Path to save final model weights (.pth)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = load_dataset(args.feasible_dir, args.infeasible_dir, seed=args.seed)
    model_cls = MODEL_REGISTRY[args.model]
    print(f'Model: {args.model}')

    run_kfold(
        model_cls=model_cls,
        dataset=dataset,
        device=device,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save_path,
    )
