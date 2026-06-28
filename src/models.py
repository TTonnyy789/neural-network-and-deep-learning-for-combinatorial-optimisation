"""
GNN Model Architectures for CTLP Feasibility Classification

All models take a torch_geometric Data object and return log-softmax
probabilities over two classes: 0 = infeasible, 1 = feasible.

Architecture families
---------------------
GCN    — Graph Convolutional Network (GraphConv layers)
GAT    — Graph Attention Network v1 (GATConv layers)
GATv2  — Graph Attention Network v2 (GATv2Conv layers)
Trans  — Transformer-style convolution (TransformerConv layers)
HEAT   — Heterogeneous Edge-type Aware Transformer (HEATConv layers)

Pooling strategies
------------------
mean   — global mean pooling
att    — attentional aggregation (learned gate network)

Naming convention: {family}_{node_features}_{pooling}
    node_features: raw = raw node features; n2v = Node2Vec embeddings
    e.g. HeatConv_raw_att_v2 = HEAT convolution, raw features, attention pooling,
         trained on the v2 (Basic) graph representation

Input dimensions depend on the graph representation:
    v2  graph: node_dim=3, edge_dim=4 (weight + 3-class one-hot)
    v3_3/v3_4/v5 graphs: node_dim=4, edge_dim=6 (weight + 5-class one-hot)
    n2v graphs: node_dim=32 or 64
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    AttentionalAggregation,
    GATConv,
    GATv2Conv,
    GlobalAttention,
    GraphConv,
    GraphNorm,
    HEATConv,
    SAGPooling,
    SetTransformerAggregation,
    TransformerConv,
    global_mean_pool,
)


# ===========================================================================
# GCN — Graph Convolutional Network
# ===========================================================================

class GCN_raw_mean(torch.nn.Module):
    """3-layer GraphConv with global mean pooling. Trained on v3_3/v3_4/v5 graphs."""

    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(4, 16, aggr='add')
        self.conv2 = GraphConv(16, 32, aggr='add')
        self.conv3 = GraphConv(32, 64, aggr='add')
        self.fc1 = torch.nn.Linear(64, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


class GCN_raw_att(torch.nn.Module):
    """3-layer GraphConv with attentional pooling. Trained on v3_3/v3_4/v5 graphs."""

    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(4, 16, aggr='mean')
        self.conv2 = GraphConv(16, 32, aggr='mean')
        self.conv3 = GraphConv(32, 64, aggr='mean')
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        ))
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.LazyBatchNorm1d(16)
        self.bn2 = torch.nn.LazyBatchNorm1d(32)
        self.bn_fc1 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index.long(), data.edge_weight
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = self.att_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GCN_n2v_att(torch.nn.Module):
    """3-layer GraphConv with attentional pooling using Node2Vec (dim=64) features."""

    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(64, 128, aggr='add')
        self.conv2 = GraphConv(128, 64, aggr='add')
        self.conv3 = GraphConv(64, 32, aggr='add')
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        ))
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


class GCN_n2v_mean(torch.nn.Module):
    """3-layer GraphConv with global mean pooling using Node2Vec (dim=64) features."""

    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(64, 128, aggr='add')
        self.conv2 = GraphConv(128, 64, aggr='add')
        self.conv3 = GraphConv(64, 32, aggr='add')
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ===========================================================================
# GAT — Graph Attention Network v1
# ===========================================================================

class GAT_raw_mean(torch.nn.Module):
    """3-layer GATConv (4 heads, concat) with global mean pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(64, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(128, 64, heads=4, concat=True, edge_dim=6)
        self.fc1 = torch.nn.Linear(256, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = F.elu(self.conv3(x, edge_index, edge_attr=edge_feature))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GAT_raw_att(torch.nn.Module):
    """3-layer GATConv (4 heads, concat) with attentional pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(64, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(128, 64, heads=4, concat=True, edge_dim=6)
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(256, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        ))
        self.fc1 = torch.nn.Linear(256, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


class GAT_n2v_mean(torch.nn.Module):
    """3-layer GATConv with mean pooling using Node2Vec (dim=32) features."""

    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(32, 64, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(256, 128, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(512, 64, heads=4, concat=True, edge_dim=6)
        self.fc1 = torch.nn.Linear(256, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn_fc1 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ===========================================================================
# GATv2 — Graph Attention Network v2
# ===========================================================================

class GATv2_raw_mean(torch.nn.Module):
    """3-layer GATv2Conv (4 heads, concat) with global mean pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATv2Conv(64, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATv2Conv(128, 64, heads=4, concat=True, edge_dim=6)
        self.fc1 = torch.nn.Linear(256, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = F.elu(self.conv3(x, edge_index, edge_attr=edge_feature))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GATv2_raw_att(torch.nn.Module):
    """7-layer GATv2Conv (4 heads, no concat) with attentional pooling.

    Deeper U-shaped architecture: 32→64→128→256→512→256→128, designed to
    capture long-range structural dependencies in hierarchical graphs.
    """

    def __init__(self):
        super().__init__()
        dims = [4, 32, 64, 128, 256, 512, 256, 128]
        convs = []
        for i in range(len(dims) - 1):
            convs.append(GATv2Conv(dims[i], dims[i + 1], heads=4, concat=False, edge_dim=6))
        self.convs = torch.nn.ModuleList(convs)
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        ))
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(d) for d in dims[1:]])
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, edge_index, edge_attr=edge_feature)))
            x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.elu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc4(x), dim=1)


class GATv2_n2v_mean(torch.nn.Module):
    """3-layer GATv2Conv with mean pooling using Node2Vec (dim=32) features."""

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(32, 64, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATv2Conv(256, 128, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATv2Conv(512, 64, heads=4, concat=True, edge_dim=6)
        self.fc1 = torch.nn.Linear(256, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn_fc1 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ===========================================================================
# TransformerConv
# ===========================================================================

class Transformer_raw_mean(torch.nn.Module):
    """7-layer TransformerConv (4 heads, concat) with global mean pooling."""

    def __init__(self):
        super().__init__()
        dims = [4, 32, 64, 128, 256, 512, 256, 128]
        self.convs = torch.nn.ModuleList([
            TransformerConv(dims[i], dims[i + 1], heads=4, concat=True, edge_dim=6)
            for i in range(len(dims) - 1)
        ])
        out_dims = [d * 4 for d in dims[1:]]
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(d) for d in out_dims])
        self.fc1 = torch.nn.Linear(out_dims[-1], 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, edge_index, edge_attr=edge_feature)))
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.elu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc4(x), dim=1)


class Transformer_raw_mean_v2(torch.nn.Module):
    """3-layer TransformerConv for Basic (v2) graphs — node_dim=3, edge_dim=4."""

    def __init__(self):
        super().__init__()
        self.conv1 = TransformerConv(3, 16, heads=4, concat=True, edge_dim=4)
        self.conv2 = TransformerConv(64, 32, heads=4, concat=True, edge_dim=4)
        self.conv3 = TransformerConv(128, 64, heads=4, concat=True, edge_dim=4)
        self.fc1 = torch.nn.Linear(256, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


class Transformer_raw_att(torch.nn.Module):
    """3-layer TransformerConv with attentional pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = TransformerConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = TransformerConv(64, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = TransformerConv(128, 64, heads=4, concat=True, edge_dim=6)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(256, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        ))
        self.fc1 = torch.nn.Linear(256, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


class Transformer_raw_att_v2(torch.nn.Module):
    """3-layer TransformerConv with attentional pooling for Basic (v2) graphs."""

    def __init__(self):
        super().__init__()
        self.conv1 = TransformerConv(3, 16, heads=4, concat=True, edge_dim=4)
        self.conv2 = TransformerConv(64, 32, heads=4, concat=True, edge_dim=4)
        self.conv3 = TransformerConv(128, 64, heads=4, concat=True, edge_dim=4)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(256, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        ))
        self.fc1 = torch.nn.Linear(256, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x, edge_index, edge_attr=edge_feature)))
        x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ===========================================================================
# HEATConv — Heterogeneous Edge-type Aware Transformer
#
# HEATConv jointly embeds node types, edge types, and edge attributes,
# making it particularly well-suited to the CTLP graphs where nodes
# (stop/vehicle/deck) and edges (next/load/unload/via/applicable) have
# distinct semantic roles.
# ===========================================================================

class HeatConv_raw_mean(torch.nn.Module):
    """7-layer HEATConv with global mean pooling for v3_3/v3_4/v5 graphs.

    num_node_types=3 (stop, vehicle, deck)
    num_edge_types=5 (next, via, load/load_via, unload, applicable)
    """

    def __init__(self):
        super().__init__()
        heat_kwargs = dict(num_node_types=3, num_edge_types=5,
                           edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)
        dims = [4, 32, 64, 128, 256, 512, 256, 128]
        self.convs = torch.nn.ModuleList([
            HEATConv(dims[i], dims[i + 1], **heat_kwargs)
            for i in range(len(dims) - 1)
        ])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(d) for d in dims[1:]])
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x = data.x
        ei, nt, et, ef = data.edge_index, data.node_type, data.edge_type, data.edge_feature
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = F.elu(self.bn_fc3(self.fc3(x)))
        return F.log_softmax(self.fc4(x), dim=1)


class HeatConv_raw_att(torch.nn.Module):
    """7-layer HEATConv with attentional pooling for v3_3/v3_4/v5 graphs.

    Uses GraphNorm (instead of BatchNorm) in conv layers for better
    stability across variable-size graphs.
    """

    def __init__(self):
        super().__init__()
        heat_kwargs = dict(num_node_types=3, num_edge_types=5,
                           edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)
        dims = [4, 32, 64, 128, 256, 512, 256, 128]
        self.convs = torch.nn.ModuleList([
            HEATConv(dims[i], dims[i + 1], **heat_kwargs)
            for i in range(len(dims) - 1)
        ])
        self.bns = torch.nn.ModuleList([GraphNorm(d) for d in dims[1:]])
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        ))
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x = data.x
        ei, nt, et, ef = data.edge_index, data.node_type, data.edge_type, data.edge_feature
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
            x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = F.elu(self.bn_fc3(self.fc3(x)))
        return F.log_softmax(self.fc4(x), dim=1)


class HeatConv_raw_mean_v2(torch.nn.Module):
    """3-layer HEATConv with global mean pooling for Basic (v2) graphs.

    num_node_types=2 (stop, vehicle)
    num_edge_types=3 (next, load, unload)
    """

    def __init__(self):
        super().__init__()
        heat_kwargs = dict(num_node_types=2, num_edge_types=3,
                           edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)
        self.conv1 = HEATConv(3, 32, **heat_kwargs)
        self.conv2 = HEATConv(32, 64, **heat_kwargs)
        self.conv3 = HEATConv(64, 128, **heat_kwargs)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)

    def forward(self, data):
        x = data.x
        ei, nt, et, ef = data.edge_index, data.node_type, data.edge_type, data.edge_feature
        x = F.elu(self.bn1(self.conv1(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
        x = F.elu(self.bn2(self.conv2(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
        x = F.elu(self.bn3(self.conv3(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class HeatConv_raw_att_v2(torch.nn.Module):
    """7-layer HEATConv with attentional pooling for Basic (v2) graphs.

    Uses GraphNorm and AttentionalAggregation for robustness on v2 graphs
    where node/edge type diversity is lower than in v3_3/v3_4/v5.
    """

    def __init__(self):
        super().__init__()
        heat_kwargs = dict(num_node_types=2, num_edge_types=3,
                           edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)
        dims = [3, 32, 64, 128, 256, 512, 256, 128]
        self.convs = torch.nn.ModuleList([
            HEATConv(dims[i], dims[i + 1], **heat_kwargs)
            for i in range(len(dims) - 1)
        ])
        self.bns = torch.nn.ModuleList([GraphNorm(d) for d in dims[1:]])
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        ))
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x = data.x
        ei, nt, et, ef = data.edge_index, data.node_type, data.edge_type, data.edge_feature
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
            x = self.dropout(x)
        x = self.att_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = F.elu(self.bn_fc3(self.fc3(x)))
        return F.log_softmax(self.fc4(x), dim=1)


class HeatConv_raw_set_v2(torch.nn.Module):
    """7-layer HEATConv with SetTransformer pooling for Basic (v2) graphs."""

    def __init__(self):
        super().__init__()
        heat_kwargs = dict(num_node_types=2, num_edge_types=3,
                           edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)
        dims = [3, 32, 64, 128, 256, 512, 256, 128]
        self.convs = torch.nn.ModuleList([
            HEATConv(dims[i], dims[i + 1], **heat_kwargs)
            for i in range(len(dims) - 1)
        ])
        self.bns = torch.nn.ModuleList([GraphNorm(d) for d in dims[1:]])
        self.set_transformer_pool = SetTransformerAggregation(
            channels=128, num_seed_points=1,
            num_encoder_blocks=1, num_decoder_blocks=1,
            heads=4, concat=True, dropout=0.1,
        )
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x = data.x
        ei, nt, et, ef = data.edge_index, data.node_type, data.edge_type, data.edge_feature
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, ei, node_type=nt, edge_type=et, edge_attr=ef)))
            x = self.dropout(x)
        x = self.set_transformer_pool(x, data.batch)
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = F.elu(self.bn_fc3(self.fc3(x)))
        return F.log_softmax(self.fc4(x), dim=1)
