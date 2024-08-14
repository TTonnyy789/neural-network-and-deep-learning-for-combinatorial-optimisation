#%%#
### Step-1 ######################################################################


## Import the necessary libraries
import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GraphConv, Sequential, GeneralConv, GATConv, GlobalAttention, Set2Set, AttentionalAggregation, SAGPooling, TopKPooling, global_add_pool, ASAPooling, GATv2Conv, TransformerConv, HEATConv, GPSConv, summary, global_sort_pool, SetTransformerAggregation, SAGPooling, global_max_pool, GraphNorm, GraphSizeNorm
import torch.nn.functional as F
# from tensorflow.keras import layers, models
# from spektral.layers import GCNConv, GlobalAvgPool
# from spektral.data import Dataset, Graph
# from spektral.data.loaders import DisjointLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import random
from scipy.sparse import coo_matrix
from torch_geometric import data as loader
import time
from graph_encoding import *
from data_preprocessing import *



### ---------------------------------------------------------------------------




#%%#
### Data ######################################################################

## Load the data from the torch file
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/feasible/raw_1M/v5/'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/infeasible/raw_1M/v5/'

feasible_data = []
infeasible_data = []

for file in os.listdir(feasible_data_dir):
    if file.endswith('.pt'):
        data = torch.load(feasible_data_dir + file)
        feasible_data.append(data)

for file in os.listdir(infeasible_data_dir):
    if file.endswith('.pt'):
        data = torch.load(infeasible_data_dir + file)
        infeasible_data.append(data)

dataset = feasible_data + infeasible_data
random.seed(42)
random.shuffle(dataset)


#%%#
### Save and Load the data ######################################################################
# ## Save the dataset
# torch.save(dataset, '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v3_3_dataset.pt')


# ## Read the data from the saved file
dataset = torch.load('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v2_dataset.pt')


### ---------------------------------------------------------------------------






#%%#
### HEAT conv essential components - Node type ########################################


node_types = []
for k in range(len(dataset)):
    cate = dataset[k].x[:, :3] ## TODO: if v2 then the [:, :2], other using [:, :3]
    node_type = torch.argmax(cate, dim=1)
    node_types.append(node_type)


## combine the node_type to the dataset
for i in range(len(dataset)):
    dataset[i].node_type = node_types[i]


### ---------------------------------------------------------------------------




#%%#
### HEAT conv essential components - Edge type ########################################

edge_types = []
for k in range(len(dataset)):
    qq = dataset[k].edge_attr
    edge_type = torch.argmax(qq, dim=1)
    edge_types.append(edge_type)


## add edge_type to the dataset
for i in range(len(dataset)):
    dataset[i].edge_type = edge_types[i]


 
### ---------------------------------------------------------------------------


## Incase the infeasible datas' y=1
# y0 = torch.tensor([0], dtype=torch.long)
# for data in infeasible_data:
#     data.y = y0


### ---------------------------------------------------------------------------




#%%#
## Change edge_feature
for i in range(len(dataset)):
    ## convert the edge_feature's first element, if value = 10, change it to 1000, and if value = 5, change it to 500, and if value = 1, change it to -1. At first, the edge_feature's first element is 10, 5, 1
    for j in range(len(dataset[i].edge_feature)):
        if dataset[i].edge_feature[j][0] == 10:
            dataset[i].edge_feature[j][0] = 18

        elif dataset[i].edge_feature[j][0] == 8:
            dataset[i].edge_feature[j][0] = 20

        if dataset[i].edge_feature[j][0] == 1:
            dataset[i].edge_feature[j][0] = 1

        elif dataset[i].edge_feature[j][0] == 0:
            dataset[i].edge_feature[j][0] = 0

        elif dataset[i].edge_feature[j][0] == 5:
            dataset[i].edge_feature[j][0] = 5


### ---------------------------------------------------------------------------



#%%#
## Change edge_weight
for i in range(len(dataset)):
    ## convert the edge_feature's first element, if value = 10, change it to 1000, and if value = 5, change it to 500, and if value = 1, change it to -1. At first, the edge_feature's first element is 10, 5, 1
    for j in range(len(dataset[i].edge_weight)):
        ## for load
        if dataset[i].edge_weight[j] == 10:
            dataset[i].edge_weight[j] = 18

        # ## for unload
        elif dataset[i].edge_weight[j] == 8:
            dataset[i].edge_weight[j] = 20

        ## for next
        if dataset[i].edge_weight[j] == 1:
            dataset[i].edge_weight[j] = 1

        ## for via
        elif dataset[i].edge_weight[j] == 0:
            dataset[i].edge_weight[j] = 0

        ## for applicable
        elif dataset[i].edge_weight[j] == 5:
            dataset[i].edge_weight[j] = 5
        

### ---------------------------------------------------------------------------




#%%#
### Models ######################################################################

## GCN for raw node feature using mean and attention
class GCN_raw_mean(torch.nn.Module):
    def __init__(self):
        super(GCN_raw_mean, self).__init__()
        self.conv1 = GraphConv(4, 16, aggr='add')
        self.conv2 = GraphConv(16, 32, aggr='add')
        self.conv3 = GraphConv(32, 64, aggr='add')


        self.fc1 = torch.nn.Linear(64, 16)  

        self.fc2 = torch.nn.Linear(64, 16)  
        self.fc3 = torch.nn.Linear(32, 16)

        self.fc4 = torch.nn.Linear(16, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(128)

        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight, edge_attr = data.x, data.edge_index, data.edge_weight, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Neural layers for label classification
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)  
        return F.log_softmax(x, dim=1) ## Log softmax for classification


class GCN_raw_att(torch.nn.Module):
    def __init__(self):
        super(GCN_raw_att, self).__init__()
        self.conv1 = GraphConv(4, 16, aggr='mean')
        self.conv2 = GraphConv(16, 32, aggr='mean')
        self.conv3 = GraphConv(32, 64, aggr='mean')

        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ))

        self.fc1 = torch.nn.Linear(64, 32)  
        self.fc2 = torch.nn.Linear(32, 16)  

        self.fc3 = torch.nn.Linear(32, 16)

        self.fc4 = torch.nn.Linear(32, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        self.bn1 = torch.nn.LazyBatchNorm1d(16)
        self.bn2 = torch.nn.LazyBatchNorm1d(32)
        self.bn3 = torch.nn.LazyBatchNorm1d(64)

        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(128)

        self.bn_fc1 = torch.nn.BatchNorm1d(32)
        self.bn_fc2 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight, edge_attr = data.x, data.edge_index, data.edge_weight, data.edge_attr

        edge_index = edge_index.long()

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = self.dropout(x)

        ## pooling
        x = self.att_pool(x, data.batch)


        ## Neural layers for label classification
        x = self.fc1(x)  
        # x = self.bn_fc1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        # x = self.fc2(x)
        # x = self.bn_fc2(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)  
        return F.log_softmax(x, dim=1) ## Log softmax for classification



class GCN_raw_att_12(torch.nn.Module):
    def __init__(self):
        super(GCN_raw_att_12, self).__init__()

        self.conv1 = GraphConv(3, 32, aggr='mean')
        self.conv2 = GraphConv(32, 64, aggr='mean')
        self.conv3 = GraphConv(64, 128, aggr='mean')
        self.conv4 = GraphConv(128, 256, aggr='mean')
        self.conv5 = GraphConv(256, 512, aggr='mean')

        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ))

        self.set_transformer_pool = SetTransformerAggregation(
            channels=512,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=4,
            concat=True,
            dropout=0.0
        )

        self.fc1 = torch.nn.Linear(512, 256)  ## Reduce to 256-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(512)

        self.bn_fc1 = torch.nn.BatchNorm1d(256)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.bn_fc3 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, edge_weight, edge_attr, batch = \
            data.x, data.edge_index, data.edge_weight, data.edge_attr, data.batch

        edge_index = edge_index.long()

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, edge_weight=edge_weight)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)

        ## Apply SAGPooling after the final GCN layer
        # x, edge_index, _, batch, _, _ = self.sag_pool(x, edge_index, None, batch=batch)

        ## Global max pooling

        # x = self.att_pool(x, batch)

        x = global_mean_pool(x, batch)

        # x = self.set_transformer_pool(x, batch)

        ## Further processing to obtain a graph-level feature
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification




## GCN for n2v node feature using mean and attention
class GCN_n2v_att(torch.nn.Module):
    def __init__(self):
        super(GCN_n2v_att, self).__init__()
        self.conv1 = GraphConv(64, 128, aggr='add')
        self.conv2 = GraphConv(128, 64, aggr='add')
        self.conv3 = GraphConv(64, 32, aggr='add')

        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        ))

        self.fc1 = torch.nn.Linear(32, 16)  
        self.fc2 = torch.nn.Linear(64, 16)  
        self.fc3 = torch.nn.Linear(32, 16)

        self.fc4 = torch.nn.Linear(16, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(128)

        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight, edge_attr = data.x, data.edge_index, data.edge_weight, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        ## Global attention pooling to aggregate node features to graph-level features
        x = self.att_pool(x, data.batch)  
        
        ## Neural layers for label classification
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)  
        return F.log_softmax(x, dim=1) ## Log softmax for classification


class GCN_n2v_mean(torch.nn.Module):
    def __init__(self):
        super(GCN_n2v_mean, self).__init__()
        self.conv1 = GraphConv(64, 128, aggr='add')
        self.conv2 = GraphConv(128, 64, aggr='add')
        self.conv3 = GraphConv(64, 32, aggr='add')

        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        ))

        self.fc1 = torch.nn.Linear(32, 16)  
        self.fc2 = torch.nn.Linear(64, 16)  
        self.fc3 = torch.nn.Linear(32, 16)

        self.fc4 = torch.nn.Linear(16, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(128)

        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_weight, edge_attr = data.x, data.edge_index, data.edge_weight, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        ## Global attention pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)   
        
        ## Neural layers for label classification
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)  
        return F.log_softmax(x, dim=1) ## Log softmax for classification



### ---------------------------------------------------------------------------



## GAT for raw node feature using mean and attention
class GAT_raw_mean(torch.nn.Module):
    def __init__(self):
        super(GAT_raw_mean, self).__init__()
        self.conv1 = GATConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(16 * 4, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        
        self.fc1 = torch.nn.Linear(64 * 4, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16 * 4)
        self.bn2 = torch.nn.BatchNorm1d(32 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        # x = self.bn3(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        # x = self.bn_fc1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)  


class GAT_n2v_mean(torch.nn.Module):
    def __init__(self):
        super(GAT_n2v_mean, self).__init__()
        self.conv1 = GATConv(32, 64, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(64 * 4, 128, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(128 * 4, 64, heads=4, concat=True, edge_dim=6)
        
        self.fc1 = torch.nn.Linear(64 * 4, 32)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(32, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(64 * 4)
        self.bn2 = torch.nn.BatchNorm1d(128 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)  


## GAT 
class GAT_raw_att(torch.nn.Module):
    def __init__(self):
        super(GAT_raw_att, self).__init__()
        self.conv1 = GATConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(16 * 4, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(64 * 4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ))
        
        self.fc1 = torch.nn.Linear(64 * 4, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16 * 4)
        self.bn2 = torch.nn.BatchNorm1d(32 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Global attention pooling to aggregate node features to graph-level features
        x = self.att_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification


class GAT_raw_att_12(torch.nn.Module):
    def __init__(self):
        super(GAT_raw_att_12, self).__init__()
        self.conv1 = GATConv(4, 32, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATConv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATConv(64 * 4, 128, heads=4, concat=True, edge_dim=6)
        self.conv4 = GATConv(128 * 4, 256, heads=4, concat=True, edge_dim=6)
        self.conv5 = GATConv(256 * 4, 512, heads=4, concat=True, edge_dim=6)

        self.set_transformer = SetTransformerAggregation(
            channels=512 * 4,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=4,
            concat=True,
            dropout=0.1
        )

        self.fc1 = torch.nn.Linear(512 * 4, 256)  ## Reduce to 256-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate

        self.bn1 = torch.nn.BatchNorm1d(32 * 4)
        self.bn2 = torch.nn.BatchNorm1d(64 * 4)
        self.bn3 = torch.nn.BatchNorm1d(128 * 4)
        self.bn4 = torch.nn.BatchNorm1d(256 * 4)
        self.bn5 = torch.nn.BatchNorm1d(512 * 4)

        self.bn_fc1 = torch.nn.BatchNorm1d(256)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.bn_fc3 = torch.nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, batch = \
            data.x, data.edge_index, data.edge_attr, data.edge_feature, data.batch

        edge_index = edge_index.long()

        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_attr=edge_feature)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, edge_attr=edge_feature)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Apply SetTransformer after the final GNN layer
        x = self.set_transformer(x, batch)

        ## Further processing to obtain a graph-level feature
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification




### ---------------------------------------------------------------------------


## GATv2 for raw node feature using mean and attention
class GATv2_raw_mean(torch.nn.Module):
    def __init__(self):
        super(GATv2_raw_mean, self).__init__()
        self.conv1 = GATv2Conv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATv2Conv(16 * 4, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATv2Conv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        
        self.fc1 = torch.nn.Linear(64 * 4, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16 * 4)
        self.bn2 = torch.nn.BatchNorm1d(32 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        # x = self.bn3(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        # x = self.bn_fc1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)


class GATv2_raw_att(torch.nn.Module):
    def __init__(self):
        super(GATv2_raw_att, self).__init__()
        self.conv1 = GATv2Conv(4, 32, heads=4, concat=False, edge_dim=6)
        self.conv2 = GATv2Conv(32, 64, heads=4, concat=False, edge_dim=6)
        self.conv3 = GATv2Conv(64, 128, heads=4, concat=False, edge_dim=6)
        self.conv4 = GATv2Conv(128, 256, heads=4, concat=False, edge_dim=6)
        self.conv5 = GATv2Conv(256, 512, heads=4, concat=False, edge_dim=6)
        self.conv6 = GATv2Conv(512, 256, heads=4, concat=False, edge_dim=6)
        self.conv7 = GATv2Conv(256, 128, heads=4, concat=False, edge_dim=6)


        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ))

        self.set_transformer_pool = SetTransformerAggregation(
            channels=128,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=4,
            concat=True,
            dropout=0.1
        )

        self.fc1 = torch.nn.Linear(128, 64)  ## Reduce to 64-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)   ## Final classification layer

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.bn6 = torch.nn.BatchNorm1d(256)
        self.bn7 = torch.nn.BatchNorm1d(128)

        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature

        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_attr=edge_feature)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, edge_attr=edge_feature)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv6(x, edge_index, edge_attr=edge_feature)
        x = self.bn6(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv7(x, edge_index, edge_attr=edge_feature)
        x = self.bn7(x)
        x = F.elu(x)
        x = self.dropout(x)


        ### Aggregation layer --------------------------------

        ## Global attention pooling to aggregate node features to graph-level features
        x = self.att_pool(x, data.batch)

        x = self.global_mean_pool(x, data.batch)

        x = self.set_transformer_pool(x, data.batch)

        ## Further processing to obtain a graph-level feature
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class GATv2_raw_att_12(torch.nn.Module):
    def __init__(self):
        super(GATv2_raw_att_12, self).__init__()
        self.conv1 = GATv2Conv(4, 32, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATv2Conv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATv2Conv(64 * 4, 128, heads=4, concat=True, edge_dim=6)
        self.conv4 = GATv2Conv(128 * 4, 256, heads=4, concat=True, edge_dim=6)
        self.conv5 = GATv2Conv(256 * 4, 512, heads=4, concat=True, edge_dim=6)

        self.sag_pool = SAGPooling(512, ratio=0.5, GNN=GATv2Conv)

        self.fc1 = torch.nn.Linear(512, 256)  ## Reduce to 256-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 2)  ## Binary classification

        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate

        self.bn1 = torch.nn.BatchNorm1d(32 * 4)
        self.bn2 = torch.nn.BatchNorm1d(64 * 4)
        self.bn3 = torch.nn.BatchNorm1d(128 * 4)
        self.bn4 = torch.nn.BatchNorm1d(256 * 4)
        self.bn5 = torch.nn.BatchNorm1d(512 * 4)

        self.bn_fc1 = torch.nn.BatchNorm1d(256)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.bn_fc3 = torch.nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, batch = \
            data.x, data.edge_index, data.edge_attr, data.edge_feature, data.batch

        edge_index = edge_index.long()

        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_attr=edge_feature)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, edge_attr=edge_feature)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Apply SAGPooling after the final GCN layer
        x, edge_index, _, batch, _, _ = self.sag_pool(x, edge_index, None, batch=batch)

        ## Global max pooling
        x = global_mean_pool(x, batch)

        ## Further processing to obtain a graph-level feature
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification



### ---------------------------------------------------------------------------


## TransformerConv for raw node feature using mean and attention
class Transformer_raw_mean(torch.nn.Module):
    def __init__(self):
        super(Transformer_raw_mean, self).__init__()
        self.conv1 = TransformerConv(4, 32, heads=4, concat=True, edge_dim=6)

        self.conv2 = TransformerConv(32 * 4, 64, heads=4, concat=True, edge_dim=6)

        self.conv3 = TransformerConv(64 * 4, 128, heads=4, concat=True, edge_dim=6)

        self.conv4 = TransformerConv(128 * 4, 256, heads=4, concat=True, edge_dim=6)

        self.conv5 = TransformerConv(256 * 4, 512, heads=4, concat=True, edge_dim=6)

        self.conv6 = TransformerConv(512 * 4, 256, heads=4, concat=True, edge_dim=6)

        self.conv7 = TransformerConv(256 * 4, 128, heads=4, concat=True, edge_dim=6)  


        self.fc1 = torch.nn.Linear (128 * 4, 64)
        self.fc2 = torch.nn. Linear (64, 32)
        self.fc3 = torch.nn. Linear (32, 16)
        self.fc4 = torch.nn. Linear (16, 2)   
            
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(32 * 4)
        self.bn2 = torch.nn.BatchNorm1d(64 * 4)
        self.bn3 = torch.nn.BatchNorm1d(128 * 4)
        self.bn4 = torch.nn.BatchNorm1d(256 * 4)
        self.bn5 = torch.nn.BatchNorm1d(512 * 4)
        self.bn6 = torch.nn.BatchNorm1d(256 * 4)
        self.bn7 = torch.nn.BatchNorm1d(128 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)


    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_attr=edge_feature)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, edge_attr=edge_feature)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv6(x, edge_index, edge_attr=edge_feature)
        x = self.bn6(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv7(x, edge_index, edge_attr=edge_feature)
        x = self.bn7(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.elu(x)
        x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification
    




## v2: for json to graph v2 weight
class Transformer_raw_mean_v2(torch.nn.Module):
    def __init__(self):
        super(Transformer_raw_mean_v2, self).__init__()
        self.conv1 = TransformerConv(3, 16, heads=4, concat=True, edge_dim=4)
        self.conv2 = TransformerConv(16 * 4, 32, heads=4, concat=True, edge_dim=4)
        self.conv3 = TransformerConv(32 * 4, 64, heads=4, concat=True, edge_dim=4)
        
        self.fc1 = torch.nn.Linear(64 * 4, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16 * 4)
        self.bn2 = torch.nn.BatchNorm1d(32 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)


class Transformer_raw_att(torch.nn.Module):
    def __init__(self):
        super(Transformer_raw_att, self).__init__()
        self.conv1 = TransformerConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = TransformerConv(16 * 4, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = TransformerConv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        
        ### ------ pooling layer ------------


        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(64 * 4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ))


        self.set_transformer_pool = SetTransformerAggregation(
            channels=64,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=4,
            concat=True,
            dropout=0.0
        )
        
        ### ------ pooling layer ------------

        self.fc1 = torch.nn.Linear(64 * 4, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16 * 4)
        self.bn2 = torch.nn.BatchNorm1d(32 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ### ------ pooling layer ------------

        ## Global attention pooling to aggregate node features to graph-level features
        x = self.att_pool(x, data.batch)  

        # x = self.set_transformer_pool(x, data.batch)
        
        ### ------ pooling layer ------------

        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification


class Transformer_raw_att_v2(torch.nn.Module):
    def __init__(self):
        super(Transformer_raw_att_v2, self).__init__()
        self.conv1 = TransformerConv(3, 16, heads=4, concat=True, edge_dim=4)
        self.conv2 = TransformerConv(16 * 4, 32, heads=4, concat=True, edge_dim=4)
        self.conv3 = TransformerConv(32 * 4, 64, heads=4, concat=True, edge_dim=4)
        
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(64 * 4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ))
        
        self.fc1 = torch.nn.Linear(64 * 4, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16 * 4)
        self.bn2 = torch.nn.BatchNorm1d(32 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Global attention pooling to aggregate node features to graph-level features
        x = self.att_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)









### ---------------------------------------------------------------------------
## HeatConv 
class HeatConv_raw_mean(torch.nn.Module):
    def __init__(self):
        super(HeatConv_raw_mean, self).__init__()

        ## edge_type_emb_dim and edge_attr_emb_dim can be any value, this is the parameter for the embedding layer for edge types

        self.conv1 = HEATConv(4, 32, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv2 = HEATConv(32, 64, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv3 = HEATConv(64, 128, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv4 = HEATConv(128, 256, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv5 = HEATConv(256, 512, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv6 = HEATConv(512, 256, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv7 = HEATConv(256, 128, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)


        
        self.fc1 = torch.nn.Linear(128, 64)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.bn6 = torch.nn.BatchNorm1d(256)
        self.bn7 = torch.nn.BatchNorm1d(128)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight

        ## If using edge_attr - one hot encoding(distinguish edge types) 
        ## If using edge_feature - one hot encoding(distinguish edge types) + edge weight

        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = \
            data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight

        x = self.conv1(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv6(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn6(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv7(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn7(x)
        x = F.elu(x)
        x = self.dropout(x)

        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        

        ## Further processing to obtain a graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.elu(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.elu(x)

        ## Final layer for label classification
        x = self.fc4(x)  
        return F.log_softmax(x, dim=1)


class HeatConv_raw_att(torch.nn.Module):
    def __init__(self):
        super(HeatConv_raw_att, self).__init__()
        
        self.conv1 = HEATConv(4, 32, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv2 = HEATConv(32, 64, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv3 = HEATConv(64, 128, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv4 = HEATConv(128, 256, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv5 = HEATConv(256, 512, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv6 = HEATConv(512, 256, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv7 = HEATConv(256, 128, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

    

        ### ------ pooling layer ------------


        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ))


        self.set_transformer_pool = SetTransformerAggregation(
            channels=128,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=4,
            concat=True,
            dropout=0.1
        )

        self.sag_pool = SAGPooling(128, ratio=0.5, GNN=GraphConv)

        # self.set2set_pool = Set2Set(64, processing_steps=3)


        ### ------ pooling layer ------------


        self.fc1 = torch.nn.Linear(128, 64)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.bn6 = torch.nn.BatchNorm1d(256)
        self.bn7 = torch.nn.BatchNorm1d(128)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(64)
        self.bn_fc2 = torch.nn.BatchNorm1d(32)
        self.bn_fc3 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = \
            data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight

        x = self.conv1(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv6(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn6(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv7(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn7(x)
        x = F.elu(x)
        x = self.dropout(x)

        

        ### ------ pooling layer ------------

        ## Global mean pooling to aggregate node features to graph-level features

        # x = self.att_pool(x, data.batch)

        x, edge_index, _, batch, _, _ = self.sag_pool(x, edge_index, None, batch=data.batch)

        x = global_mean_pool(x, batch)

        # x = self.set_transformer_pool(x, data.batch) 

        ### --------------------------------


        ## Further processing to obtain a graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.elu(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.elu(x)

        ## Final layer for label classification
        x = self.fc4(x)  
        return F.log_softmax(x, dim=1)



## v2: for json to graph v2 weight
class HeatConv_raw_mean_v2(torch.nn.Module):
    def __init__(self):
        super(HeatConv_raw_mean_v2, self).__init__()

        ## edge_attr_emb_dim can be any value, this is the parameter for the embedding layer for edge types

        self.conv1 = HEATConv(3, 32, num_node_types=2, num_edge_types=3, edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)

        self.conv2 = HEATConv(32, 64, num_node_types=2, num_edge_types=3, edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)

        self.conv3 = HEATConv(64, 128,  num_node_types=2, num_edge_types=3, edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)
        
        self.fc1 = torch.nn.Linear(128, 64)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(64, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight

        ## If using edge_attr - one hot encoding(distinguish edge types) 
        ## If using edge_feature - one hot encoding(distinguish edge types) + edge weight

        x = self.conv1(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        # x = self.bn_fc1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)


class HeatConv_raw_att_v2(torch.nn.Module):
    def __init__(self):
        super(HeatConv_raw_att_v2, self).__init__()

        ## edge_attr_emb_dim can be any value, this is the parameter for the embedding layer for edge types

        self.conv1 = HEATConv(3, 32, num_node_types=2, num_edge_types=3, edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)

        self.conv2 = HEATConv(32, 64, num_node_types=2, num_edge_types=3, edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)

        self.conv3 = HEATConv(64, 128,  num_node_types=2, num_edge_types=3, edge_type_emb_dim=3, edge_dim=4, edge_attr_emb_dim=6)
        
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ))
        
        self.fc1 = torch.nn.Linear(128, 64)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(64, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight

        ## If using edge_attr - one hot encoding(distinguish edge types) 
        ## If using edge_feature - one hot encoding(distinguish edge types) + edge weight

        x = self.conv1(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index, node_type=node_type, edge_type=edge_type, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        # x = self.dropout(x)

        ### ------ pooling layer ------------

        ## Global attention pooling to aggregate node features to graph-level features
        x = self.att_pool(x, data.batch)

        ## Global mean pooling to aggregate node features to graph-level features
        # x = global_mean_pool(x, data.batch)

        ### --------------------------------

        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)
        # x = self.bn_fc1(x)
        x = F.elu(x)
        # x = self.dropout(x)

        ## Final layer for label classification
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


### n2v --------------------------------------------------------------------------------


class GATv2_n2v_mean(torch.nn.Module):
    def __init__(self):
        super(GATv2_n2v_mean, self).__init__()
        self.conv1 = GATv2Conv(32, 64, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATv2Conv(64 * 4, 128, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATv2Conv(128 * 4, 64, heads=4, concat=True, edge_dim=6)
        
        self.fc1 = torch.nn.Linear(64 * 4, 32)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(32, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(64 * 4)
        self.bn2 = torch.nn.BatchNorm1d(128 * 4)
        self.bn3 = torch.nn.BatchNorm1d(64 * 4)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature = data.x, data.edge_index, data.edge_attr, data.edge_feature
        x = self.conv1(x, edge_index, edge_attr=edge_feature)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_feature)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_feature)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Global mean pooling to aggregate node features to graph-level features
        x = global_mean_pool(x, data.batch)  
        
        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)





### ---------------------------------------------------------------------------





#%%#
### Training ######################################################################

## Initialize model, optimizer, and loss function
# Check if MPS is available and set the device
device_1233 = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Print the device information
print(f'Using device: {device}')
print('---------------------------------------------\n')


## Create DataLoaders for training and testing sets
def create_data_loaders(train_idx, test_idx, dataset, batch_size=512):

    ## Using 20/80 split for training and testing
    train_subset = [dataset[i] for i in train_idx]
    test_subset = [dataset[i] for i in test_idx]

    train_loader = loader.DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    test_loader = loader.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

## Training and evaluation functions
def train(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y) ## Compute the loss using the true labels through CrossEntropyLoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    prediction_times = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            start_pred_time = time.time()  ## Start time for prediction
            out = model(data)
            end_pred_time = time.time()    ## End time for prediction
            pred = out.argmax(dim=1)       ## Select the class with the highest probability

            ## Time taken for each prediction
            batch_size = data.y.size(0)
            prediction_time = (end_pred_time - start_pred_time) / batch_size
            prediction_times.extend([prediction_time] * batch_size)

            correct += (pred == data.y).sum().item()
            total += batch_size
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_labels, all_preds, prediction_times


## Cross-validation, fold=5 -> 1/5 = 20% for testing, 4/5 = 80% for training
kf = KFold(n_splits=4, shuffle=True, random_state=42)


criterion = torch.nn.CrossEntropyLoss()


fold_accuracies = []
fold_train_accuracies = []
all_test_labels = []
all_test_preds = []
average_false_positive_rate = []
average_false_negative_rate = []
training_times = []
single_prediction_times = []


## Cross-validation loop and model selection
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}')

    train_loader, test_loader = create_data_loaders(train_idx, test_idx, dataset)
    

    ########### ----- Model ------------------------------------------------------------------- ###


    ##### Raw node feature representation #####
    model_12323333333 = GCN_raw_att().to(device) ## Traditional GCN with mean polling
    model_123321 = GCN_raw_mean().to(device) ## Traditional GCN with attention pooling


    model_123321 = GAT_raw_mean().to(device) ## GAT model with mean pooling 
    model_3333333 = GAT_raw_att().to(device) ## GAT model with attention pooling

    model_00101 = GAT_n2v_mean().to(device) ## GAT model with mean pooling


    model_123 = GATv2_raw_mean().to(device) ## GATv2 model with mean pooling
    model_123321 = GATv2_raw_att().to(device) ## GATv2 model with attention pooling


    model_123123 = Transformer_raw_mean().to(device) ## Transformer model with mean pooling
    model_292929229 = Transformer_raw_att().to(device) ## Transformer model with attention pooling

    model_1232312312 = Transformer_raw_mean_v2().to(device) ## Transformer model with mean pooling
    model_99999111 = Transformer_raw_att_v2().to(device) ## Transformer model with attention pooling


    model_1111111111 = HeatConv_raw_mean().to(device) ## HeatConv model with mean pooling
    model_12332 = HeatConv_raw_att().to(device) ## HeatConv model with attention pooling


    model = GCN_raw_att_12().to(device) ## GCN model with attention pooling
    model_1323 = GAT_raw_att_12().to(device) ## GAT model with attention pooling


    model_123321123 = HeatConv_raw_mean_v2().to(device) ## HeatConv model with mean pooling
    model_123321 = HeatConv_raw_att_v2().to(device) ## HeatConv model with attention pooling


    ##### Node2vec node feature #####
    model_12312312 = GATv2_n2v_mean().to(device) ## GATv2 model with mean pooling

    # model_12123123 = HEATConv_n2v_mean().to(device) ## HeatConv model with mean pooling



    ### ------ Model -------------------------------------------------------------------------- ###


    ## Print the current model name
    print('---------------------------------------------\n')
    print('Model:', model.__class__.__name__)
    print('\n---------------------------------------------\n')


    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    

    # start_time = time.time()

    for epoch in range(100):
        loss = train(model, optimizer, train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.4f}')


    # end_time = time.time()
    # training_time = end_time - start_time
    # training_times.append(training_time)

    train_accuracy, _, _, _ = evaluate(model, train_loader)


    ## Testing and evaluation with single prediction timing
    test_accuracy, test_labels, test_preds, fold_prediction_times = evaluate(model, test_loader)
    single_prediction_times.extend(fold_prediction_times)


    fold_train_accuracies.append(train_accuracy)
    fold_accuracies.append(test_accuracy)
    all_test_labels.extend(test_labels)
    all_test_preds.extend(test_preds)
    

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    # print(f'Training Time: {training_time:.2f} seconds')
    print(f'Average Prediction Time per Input: {np.mean(fold_prediction_times):.6f} seconds')
    print('---------------------------------------------\n')
    

    ## Confusion Matrix and other metrics
    cm = confusion_matrix(test_labels, test_preds)
    print('Confusion Matrix:')
    print(cm)
    average_false_positive_rate.append(cm[0][1]/(cm[0][1] + cm[0][0]))
    average_false_negative_rate.append(cm[1][0]/(cm[1][0] + cm[1][1]))
    print('---------------------------------------------\n')


    ## Visualize the confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.show()
    print('---------------------------------------------\n')



## Final outputs and statistics
print(f'Average Train Accuracy: {np.mean(fold_train_accuracies):.4f}')
print(f'Average Test Accuracy: {np.mean(fold_accuracies):.4f}')
print('-----------------------\n')
print(f'Average Training Time: {np.mean(training_times):.2f} seconds')
print(f'Average Prediction Time per Input: {np.mean(single_prediction_times):.6f} seconds')
print('-----------------------\n')
print(f'Average False Positive Rate: {np.mean(average_false_positive_rate):.4f}')
print(f'Average False Negative Rate: {np.mean(average_false_negative_rate):.4f}')
print('-----------------------\n')


## ROC Curve
fpr, tpr, _ = roc_curve(all_test_labels, all_test_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()


## Plotting the distribution of prediction times
plt.figure()
sns.boxplot(x=single_prediction_times, color='darkorange')
plt.xlabel('Prediction Time (seconds)')
plt.title('Distribution of Prediction Times per Input')
plt.show()


## Save the model
# torch.save(model.state_dict(), 'gat_model.pth')



### ---------------------------------------------------------------------------






#%%#
### Time testing data ############################################################

## Test the time from json to make a prediction using the trained model

feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/feasible'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/soft'

# infeasible_data_dir_12 = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/soft'


## Load the time testing data, excluding the soft infeasible instances
soft_infeasible = pd.read_excel('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/infeasible/soft_solution.xlsx')


soft_infeasible_instances = []

## Add the first columns in the list
for i in range(soft_infeasible.shape[0]):
    soft_infeasible_instances.append(soft_infeasible.iloc[i, 0])


### ---------------------------------------------------------------------------


#%%#
### Time testing for feasible instances ########################################

## Load the model using torch
model = torch.load('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/models/v3_3_HEAT_att.pth', map_location=torch.device('cpu'))




processing_times = []
output = []

## Select the infeasible instances excluding the soft infeasible instances, select those file with json file name, and 'solution' not in file name
infeasible_graphs = []
for file in os.listdir(infeasible_data_dir):
    if file.endswith('.json') and 'solution' not in file:
        infeasible_graphs.append(file)


## Exclude the soft infeasible instances from the infeasible instances
# infeasible_instances = [i for i in infeasible_graphs if i not in soft_infeasible_instances]

test_100 = infeasible_graphs[:100]

y0 = torch.tensor([0], dtype=torch.long)


## TODO: Before test on v3_3, remember to change the 'load via' into 'load' in the edge_att_extractor function !!!!! Verse versa for v3_4, and v5

for i in infeasible_graphs:

    start_time = time.time()
    json_data = read_json_file(os.path.join(infeasible_data_dir, i))
    graph = json_to_graph_v3_weight(json_data)

    node_features = node_feature_raw(graph)
    edge_dd = edge_index_extractor(graph)
    edge_we = edge_weight_extractor(graph)
    edge_att = edge_att_extractor(graph)

    edge_feature = torch.cat([edge_we.unsqueeze(1), edge_att], dim=1)

    node_type = torch.argmax(node_features[:, :3], dim=1)  
    edge_type = torch.argmax(edge_att, dim=1) 

    data = Data(x=node_features, edge_index=edge_dd, y=y0, edge_weight=edge_we, edge_attr=edge_att, edge_feature=edge_feature, node_type=node_type, edge_type=edge_type)
    

    try:
        out = model(data)
        pred = out.argmax(dim=1)
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        continue


    ## End time
    end_time = time.time()

    output.append(pred)

    processing_time = end_time - start_time
    processing_times.append(processing_time)


## Vilsualise the distribution of prediction times 
plt.style.use('_mpl-gallery')
plt.figure()
sns.boxplot(x=processing_times, color='darkorange')
plt.xlabel('Prediction Time (seconds)')
plt.title('Distribution of Pre-processing + Prediction Times per Input')
plt.show()


## Calculate the output accuracy
output = torch.tensor(output)
output = output.flatten()
output = output.numpy()

output_accuracy = np.sum(output == 0) / len(output)
print(f'Output Accuracy: {output_accuracy:.4f}')

### ---------------------------------------------------------------------------

## Select those output is not 0
output_1 = [i for i in output if i != 0]





# %%
