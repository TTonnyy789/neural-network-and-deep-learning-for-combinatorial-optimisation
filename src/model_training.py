#%%#
### Step-1 ######################################################################
### Change the environment into xgboost(conda)


## Import the necessary libraries
import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# import seaborn as sns
# import lightgbm as lgb
# import xgboost as xgb
# from xgboost import XGBRegressor
# from tslearn.clustering import TimeSeriesKMeans, KShape
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# from tslearn.metrics import soft_dtw, dtw
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_predict
# from nixtlats import NixtlaClient
# from nixtlats.date_features import CountryHolidays
# from pytorch_forecasting import TimeSeriesDataSet
from keras.models import Sequential, save_model, load_model, save_model
from keras.layers import Dense, LSTM, Embedding, Input
# from statsmodels.tools.eval_measures import rmse, rmspe
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1, l1_l2
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GraphConv, Sequential, GeneralConv, GATConv, GlobalAttention, Set2Set, AttentionalAggregation, SAGPooling, TopKPooling, global_add_pool, ASAPooling, GATv2Conv, TransformerConv, HEATConv, GPSConv, summary, global_sort_pool, SetTransformerAggregation
import torch.nn.functional as F
import tensorflow as tf
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
import os
# from graph_encoding import *
# from data_preprocessing import *



### ---------------------------------------------------------------------------


#%%#
### Models ######################################################################

## Load the data from the torch file
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/feasible/raw/v3_3/'
infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/infeasible/raw/v3_3/'

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
### Adjust some essential components of the dataset ########################################


### ---------------------------------------------------------------------------

node_types = []
for k in range(len(dataset)):
    cate = dataset[k].x[:, :3] ## TODO: if v2 then the [:, :2], other using [:, :3]
    node_type = torch.argmax(cate, dim=1)
    node_types.append(node_type)


## combine the node_type to the dataset
for i in range(len(dataset)):
    dataset[i].node_type = node_types[i]



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
            dataset[i].edge_feature[j][0] = 10

        elif dataset[i].edge_feature[j][0] == 8:
            dataset[i].edge_feature[j][0] = 12

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
            dataset[i].edge_weight[j] = 10

        # ## for unload
        elif dataset[i].edge_weight[j] == 8:
            dataset[i].edge_weight[j] = 12

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
        self.conv1 = GATv2Conv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = GATv2Conv(16 * 4, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = GATv2Conv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        
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


## TransformerConv for raw node feature using mean and attention
class Transformer_raw_mean(torch.nn.Module):
    def __init__(self):
        super(Transformer_raw_mean, self).__init__()
        self.conv1 = TransformerConv(4, 16, heads=4, concat=True, edge_dim=6)
        self.conv2 = TransformerConv(16 * 4, 32, heads=4, concat=True, edge_dim=6)
        self.conv3 = TransformerConv(32 * 4, 64, heads=4, concat=True, edge_dim=6)
        
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
        x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  ## Log softmax for classification



## HeatConv 
class HeatConv_raw_mean(torch.nn.Module):
    def __init__(self):
        super(HeatConv_raw_mean, self).__init__()

        ## edge_type_emb_dim and edge_attr_emb_dim can be any value, this is the parameter for the embedding layer for edge types

        self.conv1 = HEATConv(4, 32, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv2 = HEATConv(32, 64, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv3 = HEATConv(64, 128,  num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)
        
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


class HeatConv_raw_att(torch.nn.Module):
    def __init__(self):
        super(HeatConv_raw_att, self).__init__()
        self.conv1 = HEATConv(4, 16, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv2 = HEATConv(16, 32, num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)

        self.conv3 = HEATConv(32, 64,  num_node_types=3, num_edge_types=5, edge_type_emb_dim=16, edge_dim=6, edge_attr_emb_dim=16)
        

        ### ------ pooling layer ------------

        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(64, 32),
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

        # self.set2set_pool = Set2Set(64, processing_steps=3)



        ### ------ pooling layer ------------


        self.fc1 = torch.nn.Linear(64, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(64)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight
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

        ## Global mean pooling to aggregate node features to graph-level features
        # x, edge_index, _, batch, _, _ = self.pool_sag(x, edge_index, batch=data.batch) 


        x = self.set_transformer_pool(x, data.batch) 

        ### --------------------------------


        ## Further processing to obtain a 16-dimensional graph-level feature
        x = self.fc1(x)  
        # x = self.bn_fc1(x)
        x = F.elu(x)
        # x = self.dropout(x)
        
        ## Final layer for label classification
        x = self.fc2(x)  
        return F.log_softmax(x, dim=1)


class HEATConv_n2v_mean(torch.nn.Module):
    def __init__(self):
        super(HEATConv_n2v_mean, self).__init__()
        self.conv1 = HEATConv(32, 64, num_node_types=3, num_edge_types=5, edge_type_emb_dim=5, edge_dim=6, edge_attr_emb_dim=6)

        self.conv2 = HEATConv(64, 64, num_node_types=3, num_edge_types=5, edge_type_emb_dim=5, edge_dim=6, edge_attr_emb_dim=6)

        self.conv3 = HEATConv(64, 32,  num_node_types=3, num_edge_types=5, edge_type_emb_dim=5, edge_dim=6, edge_attr_emb_dim=6)
        
        self.fc1 = torch.nn.Linear(32, 16)  ## Reduce to 16-dimensional graph-level feature
        self.fc2 = torch.nn.Linear(16, 2)   
        
        self.dropout = torch.nn.Dropout(p=0.5)  ## Dropout layer with 50% dropout rate
        
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)
        
        self.bn_fc1 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index, edge_attr, edge_feature, node_type, edge_type, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_feature, data.node_type, data.edge_type, data.edge_weight
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
        
        ## Global mean pooling to aggregate node features to graph-level features
        x





#%%#
### Training ######################################################################

## Initialize model, optimizer, and loss function
device_123123 = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## Print the device information
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
print(f'Using device: {device}')
print('---------------------------------------------\n')


## Create DataLoaders for training and testing sets
def create_data_loaders(train_idx, test_idx, dataset, batch_size=128):

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
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1) ## Select the first class with the highest probability
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    return correct / total, all_labels, all_preds


## Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
## fold=5 -> 1/5 = 20% for testing, 4/5 = 80% for training

criterion = torch.nn.CrossEntropyLoss()

fold_accuracies = []
fold_train_accuracies = []
all_test_labels = []
all_test_preds = []
average_false_positive_rate = []
average_false_negative_rate = []


for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}')

    train_loader, test_loader = create_data_loaders(train_idx, test_idx, dataset)
    

    ########### ----- Model ------------------------------------------------------------------- ###


    ## Raw node feature representation
    model_090909 = GCN_raw_att().to(device) ## Traditional GCN with mean polling
    model_13123123123 = GCN_raw_mean().to(device) ## Traditional GCN with attention pooling

    model_1111 = GAT_raw_mean().to(device) ## GAT model with mean pooling 
    model_123321 = GAT_raw_att().to(device) ## GAT model with attention pooling

    model_00101 = GAT_n2v_mean().to(device) ## GAT model with mean pooling

    model_123123123 = GATv2_raw_mean().to(device) ## GATv2 model with mean pooling
    model___11231 = GATv2_raw_att().to(device) ## GATv2 model with attention pooling


    model_123123123 = Transformer_raw_mean().to(device) ## Transformer model with mean pooling
    model_000 = Transformer_raw_att().to(device) ## Transformer model with attention pooling

    model_1232312312 = Transformer_raw_mean_v2().to(device) ## Transformer model with mean pooling

    model_1231231231231 = HeatConv_raw_mean().to(device) ## HeatConv model with mean pooling
    model = HeatConv_raw_att().to(device) ## HeatConv model with attention pooling

    model_123123321321321 = HeatConv_raw_mean_v2().to(device) ## HeatConv model with mean pooling

    ## Node2vec node feature 
    model_12312312 = GATv2_n2v_mean().to(device) ## GATv2 model with mean pooling

    model_12123123 = HEATConv_n2v_mean().to(device) ## HeatConv model with mean pooling



    ### ------ Model -------------------------------------------------------------------------- ###


    ## Print the current model name
    print('---------------------------------------------\n')
    print('Model:', model.__class__.__name__)
    print('\n---------------------------------------------\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    
    for epoch in range(100):
        loss = train(model, optimizer, train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.4f}')


    train_accuracy, _, _ = evaluate(model, train_loader)
    test_accuracy, test_labels, test_preds = evaluate(model, test_loader)
    
    fold_train_accuracies.append(train_accuracy)
    fold_accuracies.append(test_accuracy)
    all_test_labels.extend(test_labels)
    all_test_preds.extend(test_preds)
    
    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('---------------------------------------------\n')
    ## Confusion Matrix for every fold
    cm = confusion_matrix(test_labels, test_preds)
    ## Print the confusion matrix
    print('Confusion Matrix: \n')
    print(cm)
    average_false_positive_rate.append(cm[0][1]/(cm[0][1] + cm[0][0]))
    average_false_negative_rate.append(cm[1][0]/(cm[1][0] + cm[1][1]))
    print('---------------------------------------------\n')
    ## Visulaize the confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.show()
    print('---------------------------------------------\n')



## Average accuracies
print(f'Average Train Accuracy: {np.mean(fold_train_accuracies):.4f}')
print(f'Average Test Accuracy: {np.mean(fold_accuracies):.4f}')
print('---------------------------------------------\n')

## Print the average false positive rate and average false negative rate
print(f'Average False Positive Rate: {np.mean(average_false_positive_rate):.4f}')
print(f'Average False Negative Rate: {np.mean(average_false_negative_rate):.4f}')
print('---------------------------------------------\n')


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



    
## Save the model
# torch.save(model.state_dict(), 'gat_model.pth')



### ---------------------------------------------------------------------------









# %%
