"""
Exploratory Data Analysis for CTLP Instance Datasets

Produces visualisation plots comparing structural properties of feasible and
soft-infeasible problem instances from the Car Transporter Loading Problem.

All plots use the 'navy' / 'darkorange' colour convention established in the
paper: navy = feasible, darkorange = infeasible.

Usage
-----
python src/eda.py \\
    --feasible-dir   data/1M_instances/feasible \\
    --infeasible-dir data/1M_instances/soft \\
    --graph-version  v5

Available plot groups (all produced by default):
    nodes_edges       — node and edge count distributions
    stop_vehicle      — stop and vehicle node count distributions
    load_distance     — vehicle stay-on-deck duration (avg and std)
    unload_count      — number of unload edges per instance
    accuracy_comparison — bar chart of paper results (Basic vs Hierarchical)
    fold_variance     — box plots of per-fold accuracy, FPR, precision, F1
    processing_time   — processing time box plots (requires pre-saved Excel files)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import sys
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import (
    read_json_file,
    json_to_graph_v5_weight,
    json_to_graph_v3_weight,
    json_to_graph_v3_4_weight,
    json_to_graph_v2_weight,
    node_extract,
    edge_extract,
    calculate_vehicle_distances,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

GRAPH_FNS = {
    'v2': json_to_graph_v2_weight,
    'v3_3': json_to_graph_v3_weight,
    'v3_4': json_to_graph_v3_4_weight,
    'v5': json_to_graph_v5_weight,
}


def _load_graphs(directory: str, graph_fn, max_files: int = None):
    """Load JSON files from a directory and convert to graphs. Skips solution files."""
    graphs, raw_data = [], []
    files = sorted(f for f in os.listdir(directory) if f.endswith('.json') and 'solution' not in f)
    if max_files:
        files = files[:max_files]
    for f in files:
        data = read_json_file(os.path.join(directory, f))
        raw_data.append(data)
        graphs.append(graph_fn(data))
    return graphs, raw_data


def load_datasets(feasible_dir: str, infeasible_dir: str, graph_version: str = 'v5',
                  max_files: int = None):
    """Load feasible and infeasible instance sets as (graphs, raw_data) tuples."""
    graph_fn = GRAPH_FNS[graph_version]
    print(f'Loading feasible instances from {feasible_dir} ...')
    feasible_graphs, feasible_raw = _load_graphs(feasible_dir, graph_fn, max_files)
    print(f'Loading infeasible instances from {infeasible_dir} ...')
    infeasible_graphs, infeasible_raw = _load_graphs(infeasible_dir, graph_fn, max_files)
    print(f'Loaded {len(feasible_graphs)} feasible + {len(infeasible_graphs)} infeasible instances')
    return feasible_graphs, feasible_raw, infeasible_graphs, infeasible_raw


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_structural_features(graphs):
    """Extract per-graph node counts, edge counts, stop and vehicle node counts."""
    n_nodes, n_edges, n_stops, n_vehicles = [], [], [], []
    for G in graphs:
        n_nodes.append(len(node_extract(G)))
        n_edges.append(len(edge_extract(G)))
        n_stops.append(sum(1 for n in G.nodes() if 'stop' in n))
        n_vehicles.append(sum(1 for n in G.nodes() if 'v' in n and 'd' not in n))
    return n_nodes, n_edges, n_stops, n_vehicles


def extract_unload_edge_counts(graphs):
    """Count unload edges per graph."""
    return [sum(1 for *_, attr in edge_extract(G) if attr['action'] == 'unload')
            for G in graphs]


def extract_stop_load_stats(graphs):
    """Compute per-graph mean and std number of vehicles loaded per stop."""
    means, stds = [], []
    for G in graphs:
        per_stop = [len(attr['load']) for _, attr in node_extract(G) if 'load' in attr]
        means.append(np.mean(per_stop) if per_stop else 0.0)
        stds.append(np.std(per_stop) if per_stop else 0.0)
    return means, stds


def extract_vehicle_distance_stats(raw_data):
    """Compute per-instance mean and std of vehicle stay-on-deck duration (stops)."""
    means, stds = [], []
    for data in raw_data:
        df = calculate_vehicle_distances(data)
        distances = df.iloc[:, 1].tolist()
        means.append(np.mean(distances))
        stds.append(np.std(distances))
    return means, stds


# ---------------------------------------------------------------------------
# Generic violin helper
# ---------------------------------------------------------------------------

def _violin_pair(data_list, x_labels, y_label, title=None, positions=None):
    """Paired violin plot with navy (feasible) / darkorange (infeasible) scheme."""
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if positions is None:
        positions = list(range(1, len(data_list) + 1))

    colors = ['navy', 'darkorange'] * (len(data_list) // 2 + 1)
    vp = ax.violinplot(data_list, showmeans=True, showmedians=True, positions=positions)
    for body, col in zip(vp['bodies'], colors):
        body.set_facecolor(col)
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    for part in ('cbars', 'cmins', 'cmaxes'):
        vp[part].set_edgecolor('black')
        vp[part].set_linewidth(1)

    midpoints = [np.mean(positions[i:i + 2]) for i in range(0, len(positions), 2)]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='y', labelsize=12)
    if title:
        ax.set_title(title, fontsize=16)

    custom_lines = [
        plt.Line2D([0], [0], color='navy', lw=4),
        plt.Line2D([0], [0], color='darkorange', lw=4),
    ]
    ax.legend(custom_lines, ['Feasible Instance', 'Infeasible Instance'],
              loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_nodes_and_edges(feasible_graphs, infeasible_graphs):
    """Violin plots: node count and edge count comparison."""
    f_nodes, f_edges, _, _ = extract_structural_features(feasible_graphs)
    i_nodes, i_edges, _, _ = extract_structural_features(infeasible_graphs)
    _violin_pair(
        [f_nodes, i_nodes, f_edges, i_edges],
        x_labels=['Number of Nodes', 'Number of Edges'],
        y_label='Count',
        positions=[1, 1.5, 3, 3.5],
    )


def plot_stop_and_vehicle_counts(feasible_graphs, infeasible_graphs):
    """Violin plots: stop node count and vehicle node count comparison."""
    _, _, f_stops, f_vehicles = extract_structural_features(feasible_graphs)
    _, _, i_stops, i_vehicles = extract_structural_features(infeasible_graphs)
    _violin_pair(
        [f_stops, i_stops, f_vehicles, i_vehicles],
        x_labels=['Number of Stops', 'Number of Vehicles'],
        y_label='Count',
        positions=[1, 1.5, 3, 3.5],
    )


def plot_vehicle_distance_stats(feasible_raw, infeasible_raw):
    """Violin plots: average and std vehicle stay duration (stops on deck)."""
    f_means, f_stds = extract_vehicle_distance_stats(feasible_raw)
    i_means, i_stds = extract_vehicle_distance_stats(infeasible_raw)
    _violin_pair(
        [f_means, i_means, f_stds, i_stds],
        x_labels=['Avg distance per vehicle', 'Std distance per vehicle'],
        y_label='Distance (stops)',
        positions=[1, 1.5, 3, 3.5],
    )


def plot_unload_edge_counts(feasible_graphs, infeasible_graphs):
    """Violin plot: number of unload edges per instance."""
    f_unload = extract_unload_edge_counts(feasible_graphs)
    i_unload = extract_unload_edge_counts(infeasible_graphs)
    _violin_pair(
        [f_unload, i_unload],
        x_labels=['Unloading edges'],
        y_label='Count',
        positions=[1, 1.5],
    )


def plot_accuracy_comparison():
    """Bar chart comparing cross-validated accuracy metrics for Basic vs Hierarchical.

    Values taken directly from the paper's results table.
    """
    data = {
        'Basic': {
            'Accuracy': 0.85353769, 'Precision': 0.87902097, 'Recall': 0.82039439,
            'F1 Score': 0.84830046, 'FPR': 0.1133573, 'FNR': 0.1796066,
        },
        'Hierarchical': {
            'Accuracy': 0.86221323, 'Precision': 0.85665691, 'Recall': 0.87024394,
            'F1 Score': 0.86328133, 'FPR': 0.14572944, 'FNR': 0.12975064,
        },
    }
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'FPR', 'FNR']
    y_limits = {
        'Accuracy': (0.7, 0.9), 'Precision': (0.7, 0.9), 'Recall': (0.7, 0.9),
        'F1 Score': (0.7, 0.9), 'FPR': (0.1, 0.2), 'FNR': (0.1, 0.3),
    }
    representations = list(data.keys())

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    for ax, metric in zip(axs.flatten(), metrics):
        values = [data[r][metric] for r in representations]
        ax.bar(representations, values, color='navy')
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim(y_limits[metric])
        for j, val in enumerate(values):
            ax.text(j, val + 0.002, f'{val:.4f}', ha='center', va='bottom')
        ax.set_ylabel(metric)
        ax.set_xlabel('Graph Representation')
    plt.tight_layout()
    plt.show()


def plot_fold_variance():
    """Box plots showing per-fold variance in accuracy, FPR, precision, and F1."""
    accuracy_data = {
        'Basic': [0.850974, 0.847889, 0.859649, 0.855408, 0.845961,
                  0.849431, 0.854444, 0.854058, 0.861963, 0.855601],
        'Hierarchical': [0.866397, 0.861926, 0.863312, 0.855408, 0.864276,
                         0.862541, 0.867746, 0.858492, 0.860806, 0.861191],
    }
    fpr_data = {
        'Basic': [0.112779, 0.099391, 0.104758, 0.124905, 0.097532,
                  0.099726, 0.127349, 0.102033, 0.119420, 0.145685],
        'Hierarchical': [0.147421, 0.160700, 0.149270, 0.135915, 0.161770,
                         0.132963, 0.154583, 0.147297, 0.143457, 0.123911],
    }
    precision_data = {
        'Basic': [0.878018, 0.886225, 0.886202, 0.866315, 0.893207,
                  0.891903, 0.866613, 0.887049, 0.873586, 0.861027],
        'Hierarchical': [0.856132, 0.843064, 0.853207, 0.857880, 0.850145,
                         0.860977, 0.850740, 0.853097, 0.855162, 0.878068],
    }
    f1_data = {
        'Basic': [0.845121, 0.837487, 0.853815, 0.850419, 0.839138,
                  0.843456, 0.851055, 0.846606, 0.857993, 0.858919],
        'Hierarchical': [0.868025, 0.863619, 0.864462, 0.852129, 0.869388,
                         0.863593, 0.870076, 0.858683, 0.860132, 0.862280],
    }

    sns.set(style='whitegrid')
    for df_data, ylabel, ylim, title in [
        (accuracy_data, 'Accuracy', (0.725, 0.915),
         'Accuracy Variance Across Folds'),
        (fpr_data, 'FPR', (0, 0.4),
         'FPR Variance Across Folds'),
        (precision_data, 'Precision', (0.775, 0.95),
         'Precision Variance Across Folds'),
        (f1_data, 'F1 Score', (0.775, 0.9),
         'F1 Score Variance Across Folds'),
    ]:
        df = pd.DataFrame(df_data)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, palette='Set2', showmeans=False)
        sns.stripplot(data=df, palette='Set2', jitter=False, size=8,
                      linewidth=1, edgecolor='gray')
        plt.ylabel(ylabel, fontsize=18)
        plt.title(title, fontsize=16)
        plt.ylim(ylim)
        plt.xticks(ticks=range(len(df.columns)), labels=df.columns, fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_roc_and_pr_curves(file_paths: list, labels: list, colors: list):
    """Plot ROC and Precision-Recall curves from saved .npz files.

    Each .npz must contain 'true_labels' and 'pred_probs' arrays
    (generated by src/evaluate.py --save-roc).

    Parameters
    ----------
    file_paths : paths to .npz files from evaluate.py
    labels     : legend labels, one per file
    colors     : line colours, one per file

    Example
    -------
    plot_roc_and_pr_curves(
        file_paths=['data/processed/performance_matrix/roc_pr_data_v2.npz',
                    'data/processed/performance_matrix/roc_pr_data_v5.npz'],
        labels=['Basic', 'Hierarchical'],
        colors=['teal', 'pink'],
    )
    """
    roc_curves, pr_curves, roc_aucs, pr_aucs = [], [], [], []

    for path in file_paths:
        d = np.load(path, allow_pickle=True)
        true_labels = d['true_labels']
        pred_probs = d['pred_probs']
        d.close()

        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))
        roc_aucs.append(roc_auc)

        precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
        pr_auc = auc(recall, precision)
        pr_curves.append((precision, recall))
        pr_aucs.append(pr_auc)

    plt.figure(figsize=(10, 8))
    for (fpr, tpr), label, col, roc_auc in zip(roc_curves, labels, colors, roc_aucs):
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})', color=col, lw=2)
    plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    for (precision, recall), label, col, pr_auc in zip(pr_curves, labels, colors, pr_aucs):
        plt.plot(recall, precision, label=f'{label} (AUC = {pr_auc:.2f})', color=col, lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_processing_times(inference_xlsx: str, total_xlsx: str):
    """Box plots of inference and total processing times across graph representations.

    The Excel files are generated by src/evaluate.py and should have columns
    named 'v2', 'v3_3', 'v3_4', 'v5'.

    Parameters
    ----------
    inference_xlsx : path to processing_times_*_inference.xlsx
    total_xlsx     : path to processing_times_*_total.xlsx
    """
    for path, ylabel, ylim, title in [
        (inference_xlsx, 'Processing Time (seconds)', (0, 0.6),
         'Inference Time per Graph Representation'),
        (total_xlsx, 'Processing Time (seconds)', (0, 0.05),
         'Total (Pre-processing + Inference) Time per Graph Representation'),
    ]:
        df = pd.read_excel(path)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.boxplot(
            [df['v2'], df['v3_3'], df['v3_4'], df['v5']],
            showmeans=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='navy'),
            medianprops=dict(color='navy'),
            meanprops=dict(marker='o', markerfacecolor='darkorange', markeredgecolor='black'),
        )
        means = [df[col].mean() for col in ['v2', 'v3_3', 'v3_4', 'v5']]
        for i, mean in enumerate(means, start=1):
            ax.text(i + 0.03, mean + ylim[1] * 0.02, f'{mean:.4f}',
                    ha='left', va='bottom', fontsize=13)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Basic', 'Deck Assign', 'Deck Co-use', 'Hierarchical'], fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_ylim(ylim)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='EDA plots for CTLP instance datasets')
    p.add_argument('--feasible-dir', required=True)
    p.add_argument('--infeasible-dir', required=True)
    p.add_argument('--graph-version', default='v5', choices=list(GRAPH_FNS.keys()))
    p.add_argument('--max-files', type=int, default=None,
                   help='Cap files loaded per directory (for quick iteration)')
    p.add_argument('--plots', nargs='+',
                   default=['nodes_edges', 'stop_vehicle', 'load_distance',
                            'unload_count', 'accuracy_comparison', 'fold_variance'],
                   choices=['nodes_edges', 'stop_vehicle', 'load_distance',
                            'unload_count', 'accuracy_comparison', 'fold_variance'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    plot_set = set(args.plots)
    needs_graphs = plot_set & {'nodes_edges', 'stop_vehicle', 'load_distance', 'unload_count'}

    feasible_graphs, feasible_raw, infeasible_graphs, infeasible_raw = [], [], [], []

    if needs_graphs:
        feasible_graphs, feasible_raw, infeasible_graphs, infeasible_raw = load_datasets(
            args.feasible_dir, args.infeasible_dir,
            graph_version=args.graph_version,
            max_files=args.max_files,
        )

    if 'nodes_edges' in plot_set:
        plot_nodes_and_edges(feasible_graphs, infeasible_graphs)

    if 'stop_vehicle' in plot_set:
        plot_stop_and_vehicle_counts(feasible_graphs, infeasible_graphs)

    if 'load_distance' in plot_set:
        plot_vehicle_distance_stats(feasible_raw, infeasible_raw)

    if 'unload_count' in plot_set:
        plot_unload_edge_counts(feasible_graphs, infeasible_graphs)

    if 'accuracy_comparison' in plot_set:
        plot_accuracy_comparison()

    if 'fold_variance' in plot_set:
        plot_fold_variance()
