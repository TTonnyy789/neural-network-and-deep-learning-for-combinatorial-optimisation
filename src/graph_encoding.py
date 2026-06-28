"""
Graph Feature Encoding for CTLP GNN Models

Converts NetworkX graphs (produced by data_preprocessing.py) into PyTorch
tensors suitable for torch_geometric Data objects.

Node features
-------------
Three variants are available depending on the graph representation:

    node_feature_raw_v2   — [is_stop, is_vehicle, representation]          (v2)
    node_feature_raw      — [is_stop, is_vehicle, is_deck, representation]  (v3_3/v3_4)
    node_feature_raw_v5   — [is_stop, is_vehicle, is_deck, representation]  (v5)

Edge features
-------------
One-hot vectors encoding the action type, optionally combined with edge weight:

    edge_att_extractor_v2 — 3-class: [next, load, unload]            (v2)
    edge_att_extractor_v3 — 5-class: [next, via, load, unload, applicable]     (v3_3)
    edge_att_extractor    — 5-class: [next, via, load via, unload, applicable]  (v3_4/v5)
    edge_feature_extractor — concatenates edge weight with 5-class one-hot

Node embeddings (alternative to raw features)
----------------------------------------------
    node_feature_node2vec — 32-dim Node2Vec embeddings (requires training)
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec as Node2Vec_pyg
from tqdm.notebook import tqdm

from data_preprocessing import node_extract, edge_extract


# ---------------------------------------------------------------------------
# Index / weight extractors
# ---------------------------------------------------------------------------

def edge_index_extractor(G: nx.MultiDiGraph) -> torch.Tensor:
    """Return edge index tensor of shape [2, num_edges] (COO format)."""
    A = nx.adjacency_matrix(G, weight='weight').tocoo()
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    return torch.stack([row, col], dim=0)


def edge_weight_extractor(G: nx.MultiDiGraph) -> torch.Tensor:
    """Return edge weight tensor of shape [num_edges]."""
    A = nx.adjacency_matrix(G, weight='weight').tocoo()
    return torch.tensor(A.data, dtype=torch.float)


# ---------------------------------------------------------------------------
# Edge attribute extractors (one-hot action encoding)
# ---------------------------------------------------------------------------

def edge_att_extractor_v2(G: nx.MultiDiGraph) -> torch.Tensor:
    """3-class one-hot for Basic (v2) graphs: [next, load, unload]."""
    action_map = {'next': [1, 0, 0], 'load': [0, 1, 0], 'unload': [0, 0, 1]}
    edges = edge_extract(G)
    features = [action_map.get(e[2]['action'], [0, 0, 0]) for e in edges]
    return torch.tensor(features, dtype=torch.float32)


def edge_att_extractor_v3(G: nx.MultiDiGraph) -> torch.Tensor:
    """5-class one-hot for Deck Assign (v3_3) graphs.

    Classes: [next, via, load, unload, applicable]
    Note: v3_3 uses 'load' (not 'load via') for stop→vehicle edges.
    """
    action_map = {
        'next':       [1, 0, 0, 0, 0],
        'via':        [0, 1, 0, 0, 0],
        'load':       [0, 0, 1, 0, 0],
        'unload':     [0, 0, 0, 1, 0],
        'applicable': [0, 0, 0, 0, 1],
    }
    edges = edge_extract(G)
    features = [action_map.get(e[2]['action'], [0, 0, 0, 0, 0]) for e in edges]
    return torch.tensor(features, dtype=torch.float32)


def edge_att_extractor(G: nx.MultiDiGraph) -> torch.Tensor:
    """5-class one-hot for Deck Co-use (v3_4) and Hierarchical (v5) graphs.

    Classes: [next, via, load via, unload, applicable]
    Note: v3_4/v5 use 'load via' for stop→deck edges.
    """
    action_map = {
        'next':       [1, 0, 0, 0, 0],
        'via':        [0, 1, 0, 0, 0],
        'load via':   [0, 0, 1, 0, 0],
        'unload':     [0, 0, 0, 1, 0],
        'applicable': [0, 0, 0, 0, 1],
    }
    edges = edge_extract(G)
    features = [action_map.get(e[2]['action'], [0, 0, 0, 0, 0]) for e in edges]
    return torch.tensor(features, dtype=torch.float32)


def edge_feature_extractor(G: nx.MultiDiGraph) -> torch.Tensor:
    """Concatenate [edge_weight | 5-class one-hot] → shape [num_edges, 6].

    Used with v3_4 and v5 graphs where 'load via' is the stop→deck action.
    """
    weights = edge_weight_extractor(G).unsqueeze(1)
    attrs = edge_att_extractor(G)
    return torch.cat([weights, attrs], dim=1)


# ---------------------------------------------------------------------------
# Node feature extractors
# ---------------------------------------------------------------------------

def node_feature_raw_v2(graph: nx.MultiDiGraph,
                         feature: str = 'representation') -> torch.Tensor:
    """2-class one-hot node type + scalar feature for Basic (v2) graphs.

    Returns tensor of shape [num_nodes, 3]:
        [is_stop, is_vehicle, representation]
    """
    nodes = node_extract(graph)
    features = []
    for name, attr in nodes:
        val = attr.get(feature, 0)
        if 'stop' in name:
            features.append([1, 0, val])
        else:
            features.append([0, 1, val])
    return torch.tensor(features, dtype=torch.float32)


def node_feature_raw(graph: nx.MultiDiGraph,
                      feature: str = 'representation') -> torch.Tensor:
    """3-class one-hot node type + scalar feature for v3_3/v3_4 graphs.

    Returns tensor of shape [num_nodes, 4]:
        [is_stop, is_vehicle, is_deck, representation]

    Node type detection:
        stop_i  → is_stop
        v_k     → is_vehicle  (contains 'v', no 'd')
        d_j     → is_deck     (contains 'd', no 'v')
    """
    nodes = node_extract(graph)
    features = []
    for name, attr in nodes:
        val = attr.get(feature, 0)
        if 'stop' in name:
            features.append([1, 0, 0, val])
        elif 'v' in name and 'd' not in name:
            features.append([0, 1, 0, val])
        else:
            features.append([0, 0, 1, val])
    return torch.tensor(features, dtype=torch.float32)


def node_feature_raw_v5(graph: nx.MultiDiGraph,
                         feature: str = 'representation') -> torch.Tensor:
    """3-class one-hot node type + scalar feature for Hierarchical (v5) graphs.

    Returns tensor of shape [num_nodes, 4]:
        [is_stop, is_vehicle, is_deck_node, representation]

    Node type detection for v5 (where deck nodes contain both 'd' and 'v'):
        stop_i    → is_stop
        v_k       → is_vehicle  (contains 'v', no 'd')
        d_j v_k   → is_deck_node (contains 'v' and 'd')
    """
    nodes = node_extract(graph)
    features = []
    for name, attr in nodes:
        val = attr.get(feature, 0)
        if 'stop' in name:
            features.append([1, 0, 0, val])
        elif 'v' in name and 'd' not in name:
            features.append([0, 1, 0, val])
        else:
            features.append([0, 0, 1, val])
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Node2Vec embeddings (alternative to raw features)
# ---------------------------------------------------------------------------

def node_feature_node2vec(graph: nx.MultiDiGraph,
                           embedding_dim: int = 32,
                           walk_length: int = 40,
                           context_size: int = 15,
                           walks_per_node: int = 25,
                           epochs: int = 70,
                           device: str = 'cpu') -> torch.Tensor:
    """Train a Node2Vec model on a single graph and return node embeddings.

    Returns tensor of shape [num_nodes, embedding_dim].
    Training is done per-graph; for large datasets prefer pre-training on
    a combined graph and passing the model externally.
    """
    data = from_networkx(graph)
    model = Node2Vec_pyg(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=1, q=1,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch:02d} | Loss: {total_loss / len(loader):.4f}')

    embeddings = model(torch.arange(data.num_nodes, device=device)).detach().cpu()
    return embeddings.float()
