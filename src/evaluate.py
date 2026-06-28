"""
Evaluation Script for Trained CTLP GNN Models

Loads a saved model (.pth) and a directory of pre-processed .pt graph files,
runs inference, and reports:
  - Accuracy, Precision, Recall, F1-score
  - False Positive Rate and False Negative Rate
  - ROC-AUC and Precision-Recall AUC
  - Confusion matrix (printed + plotted)
  - Distribution of per-sample prediction times

Usage
-----
python src/evaluate.py \\
    --model-path models/v5_HEAT_att_late.pth \\
    --feasible-dir  data/1M_instances/feasible/ \\
    --infeasible-dir data/1M_instances/soft/ \\
    --graph-version v5 \\
    --save-roc data/processed/performance_matrix/roc_pr_data_v5.npz

--graph-version controls which feature extractor is used:
    v2        — Basic graph (node_dim=3, edge_dim=4)
    v3_3      — Deck Assign graph (node_dim=4, edge_dim=6, 'load' edge type)
    v3_4      — Deck Co-use graph (node_dim=4, edge_dim=6, 'load via' edge type)
    v5        — Hierarchical graph (node_dim=4, edge_dim=6, per-vehicle deck trees)
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data
from torch_geometric import data as loader

import sys
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import (
    read_json_file,
    json_to_graph_v2_weight,
    json_to_graph_v3_weight,
    json_to_graph_v3_4_weight,
    json_to_graph_v5_weight,
)
from graph_encoding import (
    edge_index_extractor,
    edge_weight_extractor,
    edge_att_extractor_v2,
    edge_att_extractor_v3,
    edge_att_extractor,
    node_feature_raw_v2,
    node_feature_raw,
    node_feature_raw_v5,
)


# ---------------------------------------------------------------------------
# Graph version configuration
# ---------------------------------------------------------------------------

GRAPH_CONFIGS = {
    'v2': {
        'graph_fn': json_to_graph_v2_weight,
        'node_fn': node_feature_raw_v2,
        'edge_att_fn': edge_att_extractor_v2,
        'n_node_types': 2,
    },
    'v3_3': {
        'graph_fn': json_to_graph_v3_weight,
        'node_fn': node_feature_raw,
        'edge_att_fn': edge_att_extractor_v3,
        'n_node_types': 3,
    },
    'v3_4': {
        'graph_fn': json_to_graph_v3_4_weight,
        'node_fn': node_feature_raw,
        'edge_att_fn': edge_att_extractor,
        'n_node_types': 3,
    },
    'v5': {
        'graph_fn': json_to_graph_v5_weight,
        'node_fn': node_feature_raw_v5,
        'edge_att_fn': edge_att_extractor,
        'n_node_types': 3,
    },
}


# ---------------------------------------------------------------------------
# Data preparation from raw JSON files
# ---------------------------------------------------------------------------

def json_to_data(json_path: str, label: int, cfg: dict) -> Data | None:
    """Convert a raw JSON instance to a torch_geometric Data object."""
    try:
        raw = read_json_file(json_path)
        G = cfg['graph_fn'](raw)
        node_features = cfg['node_fn'](G)
        edge_idx = edge_index_extractor(G)
        edge_w = edge_weight_extractor(G)
        edge_att = cfg['edge_att_fn'](G)
        edge_feature = torch.cat([edge_w.unsqueeze(1), edge_att], dim=1)
        n_types = cfg['n_node_types']
        node_type = torch.argmax(node_features[:, :n_types], dim=1)
        edge_type = torch.argmax(edge_att, dim=1)
        y = torch.tensor([label], dtype=torch.long)
        return Data(
            x=node_features,
            edge_index=edge_idx,
            y=y,
            edge_weight=edge_w,
            edge_attr=edge_att,
            edge_feature=edge_feature,
            node_type=node_type,
            edge_type=edge_type,
        )
    except Exception as e:
        print(f'Warning: skipping {json_path}: {e}')
        return None


def load_instances(feasible_dir: str, infeasible_dir: str, cfg: dict):
    """Load all JSON instances from feasible and infeasible directories."""
    dataset, labels = [], []

    for f in sorted(os.listdir(feasible_dir)):
        if f.endswith('.json') and 'solution' not in f:
            d = json_to_data(os.path.join(feasible_dir, f), label=1, cfg=cfg)
            if d is not None:
                dataset.append(d)
                labels.append(1)

    for f in sorted(os.listdir(infeasible_dir)):
        if f.endswith('.json') and 'solution' not in f:
            d = json_to_data(os.path.join(infeasible_dir, f), label=0, cfg=cfg)
            if d is not None:
                dataset.append(d)
                labels.append(0)

    print(f'Loaded {sum(l == 1 for l in labels)} feasible + '
          f'{sum(l == 0 for l in labels)} infeasible instances')
    return dataset


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataset, batch_size: int = 512, device: str = 'cpu'):
    """Run inference over dataset; return (labels, preds, pred_probs, pred_times)."""
    model.eval()
    data_loader = loader.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_labels, all_preds, all_probs, pred_times = [], [], [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            t0 = time.time()
            out = model(data)
            t1 = time.time()

            probs = torch.exp(out)[:, 1]
            preds = out.argmax(dim=1)
            batch_n = data.y.size(0)
            per_sample = (t1 - t0) / batch_n

            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            pred_times.extend([per_sample] * batch_n)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(pred_times),
    )


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------

def print_metrics(labels, preds, probs):
    """Print classification metrics to stdout."""
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn)
    fnr_val = fn / (fn + tp)
    roc_auc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    print('\n=== Evaluation Results ===')
    print(classification_report(labels, preds, target_names=['Infeasible', 'Feasible']))
    print(f'False Positive Rate (FPR): {fpr_val:.4f}')
    print(f'False Negative Rate (FNR): {fnr_val:.4f}')
    print(f'ROC-AUC : {roc_auc:.4f}')
    print(f'PR-AUC  : {pr_auc:.4f}')
    print(f'\nConfusion Matrix:\n{cm}')
    return cm, fpr_val, fnr_val, roc_auc, pr_auc


def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Infeasible', 'Feasible'],
                yticklabels=['Infeasible', 'Feasible'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_pr_curve(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='teal', lw=2, label=f'PR-AUC = {pr_auc:.2f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def plot_prediction_times(pred_times):
    plt.figure()
    sns.boxplot(x=pred_times, color='darkorange')
    plt.xlabel('Prediction time (seconds)', fontsize=14)
    plt.title('Distribution of Per-sample Prediction Times')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a saved CTLP GNN model')
    p.add_argument('--model-path', required=True, help='Path to saved .pth model')
    p.add_argument('--feasible-dir', required=True)
    p.add_argument('--infeasible-dir', required=True)
    p.add_argument('--graph-version', default='v5', choices=list(GRAPH_CONFIGS.keys()),
                   help='Graph representation used when training (default: v5)')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--save-roc', default=None,
                   help='Save ROC/PR curve data to a .npz file for later plotting')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = torch.load(args.model_path, map_location=device)
    model.eval()
    print(f'Loaded model: {model.__class__.__name__}')

    cfg = GRAPH_CONFIGS[args.graph_version]
    dataset = load_instances(args.feasible_dir, args.infeasible_dir, cfg)

    labels, preds, probs, pred_times = run_inference(
        model, dataset, batch_size=args.batch_size, device=device
    )

    cm, fpr_val, fnr_val, roc_auc, pr_auc = print_metrics(labels, preds, probs)

    print(f'\nAvg prediction time per sample: {np.mean(pred_times):.6f}s')

    plot_confusion_matrix(cm)
    plot_roc_curve(labels, probs)
    plot_pr_curve(labels, probs)
    plot_prediction_times(pred_times)

    if args.save_roc:
        os.makedirs(os.path.dirname(args.save_roc), exist_ok=True)
        np.savez(args.save_roc, true_labels=labels, pred_probs=probs)
        print(f'ROC/PR data saved to {args.save_roc}')
