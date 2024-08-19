#%%#
### Step-1 ######################################################################
### Read the json file

import os
import json
import pickle
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import networkx as nx
import numpy as np
import torch
from node2vec import Node2Vec
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec as Node2Vec_2
from torch_geometric.utils import from_networkx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.optim import Adam
from karateclub import FeatherGraph
from sklearn.model_selection import train_test_split
from node2vec.edges import HadamardEmbedder
from torch_geometric.nn import GCNConv, global_mean_pool
from data_preprocessing import *


data_1 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/7CT_s5_v7-7766.json')
data_2 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/7CT_s6_v6-9122.json')
data_3 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/11CT_s5_v13-6106.json')
data_4 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/11CT_s12_v34-6729.json')


## Print the data form the json file 
print("-------------------------------------------------")
print("Data from the json file")
print("-------------------------------------------------")
print(data_1)
print("------------------------------------------------- \n")
print(data_2)
print("------------------------------------------------- \n")
print(data_3)
print("------------------------------------------------- \n")
print(data_4)


### ---------------------------------------------------------------------------



#%%#
### Step-2 ######################################################################
### Load the example instances


## Prepare graph data
G_1 = json_to_graph(data_1)
GG_1 = json_to_graph_v2(data_1)
GGG_1 = json_to_graph_v3(data_1)
GGGG_1 = json_to_graph_v4(data_1)
# F_1 = json_to_graph_v5(data_1)
G_2 = json_to_graph(data_2)
G_3 = json_to_graph(data_3)
G_4 = json_to_graph(data_4)
GG_4 = json_to_graph_v2(data_4)
GGG_4 = json_to_graph_v3(data_4)
GGGG_4 = json_to_graph_v4(data_4)
# F_4 = json_to_graph_v5(data_4)


## Print the graph
print("-------------------------------------------------")
print("Graph from the json data")
print("-------------------------------------------------")
print(G_1.nodes(data=True))
print("\n")
print(G_1.edges(data=True))
print("------------------------------------------------- \n")
print(G_2.nodes(data=True))
print("\n")
print(G_2.edges(data=True))
print("------------------------------------------------- \n")
print(G_3.nodes(data=True))
print("\n")
print(G_3.edges(data=True))
print("------------------------------------------------- \n")
print(G_4.nodes(data=True))
print("\n")
print(G_4.edges(data=True))
print("------------------------------------------------- \n")

## Draw the graph
# print("json to graph v1 - graph 1")
# plt.figure(figsize=(10, 10))
# nx.draw(G_1, with_labels=True, font_weight='bold')
# plt.show()
# print("------------------------------------------------- \n")
# print("json to graph v1 - graph 4")
# plt.figure(figsize=(10, 10))
# nx.draw(G_4, with_labels=True, font_weight='bold')
# plt.show()
# print("------------------------------------------------- \n")
# print("json to graph v2 - graph 1")
# plt.figure(figsize=(10, 10))
# nx.draw(GG_1, with_labels=True, font_weight='bold')
# plt.show()
# print("------------------------------------------------- \n")
# print("json to graph v2 - graph 4")
# plt.figure(figsize=(10, 10))
# nx.draw(GG_4, with_labels=True, font_weight='bold')
# plt.show()
print("------------------------------------------------- \n")
print("json to graph v3 - graph 1")
plt.figure(figsize=(10, 10))
nx.draw(GGG_1, with_labels=True, font_weight='bold')
plt.show()
print("------------------------------------------------- \n")
print("json to graph v3 - graph 4")
plt.figure(figsize=(10, 10))
nx.draw(GGG_4, with_labels=True, font_weight='bold')
plt.show()
print("------------------------------------------------- \n")
print("json to graph v4 - graph 1")
plt.figure(figsize=(10, 10))
nx.draw(GGGG_1, with_labels=True, font_weight='bold')
plt.show()
print("------------------------------------------------- \n")
print("json to graph v4 - graph 4")
plt.figure(figsize=(10, 10))
nx.draw(GGGG_4, with_labels=True, font_weight='bold')
plt.show()
print("------------------------------------------------- \n")



### ---------------------------------------------------------------------------


#%%#
### Similar graphs selection ####################################################

## Basic graph information extraction
def adjacency_metric_extract(G):
    A = nx.adjacency_matrix(G, weight='weight')
    return A


## Extract the nodes from the graph
def node_extract(G):
    nodes = []
    for node in G.nodes(data=True):
        nodes.append(node)
    return nodes


## Extract the edges from the graph
def edge_extract(G):
    edges = []
    for edge in G.edges(data=True):
        edges.append(edge)
    return edges


## Edge_index matrix using torch_geometric.node2vec
def edge_index_extractor(G):
    # Extract the adjacency matrix in COO format
    A = nx.adjacency_matrix(G, weight='weight').tocoo()
    
    # Extract row, column, and data (weight) from the COO matrix
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    
    # Extract edge weights
    edge_weight = torch.tensor(A.data, dtype=torch.float)
    
    return edge_index


## Edge_weight matrix using torch_geometric.node2vec
def edge_weight_extractor(G):
    a = nx.adjacency_matrix(G, weight='weight')
    coo_matrix = a.tocoo()
    edge_weight = torch.tensor(coo_matrix.data, dtype=torch.float)
    return edge_weight


## Edge_weight matrix using torch_geometric.node2vec one hot encoding for edge attribute
## TODO: v3_3 -> load, v3_4 and v5 -> load via !!!!!!!!!!!!!!!
def edge_att_extractor(G):
    ## For attribute in edge, 'action' doing one-hot encoding, there are 'next', 'via', 'load', 'unload', 'applicable'
    ## Build a tensor matrix for the edge feature, store the one-hot encoding for each edge
    edge_features = []
    for edge in range(len(edge_extract(G))):
        if edge_extract(G)[edge][2]['action'] == 'next':
            edge_features.append([1, 0, 0, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'via':
            edge_features.append([0, 1, 0, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'load via': ## v5, v3_4=load via, v3_3=load !!!!!!
            edge_features.append([0, 0, 1, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'unload':
            edge_features.append([0, 0, 0, 1, 0])
        elif edge_extract(G)[edge][2]['action'] == 'applicable':
            edge_features.append([0, 0, 0, 0, 1])

    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    return edge_features


def edge_att_extractor_v3(G):
    ## For attribute in edge, 'action' doing one-hot encoding, there are 'next', 'via', 'load', 'unload', 'applicable'
    ## Build a tensor matrix for the edge feature, store the one-hot encoding for each edge
    edge_features = []
    for edge in range(len(edge_extract(G))):
        if edge_extract(G)[edge][2]['action'] == 'next':
            edge_features.append([1, 0, 0, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'via':
            edge_features.append([0, 1, 0, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'load': ## v5, v3_4=load via, v3_3=load !!!!!!
            edge_features.append([0, 0, 1, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'unload':
            edge_features.append([0, 0, 0, 1, 0])
        elif edge_extract(G)[edge][2]['action'] == 'applicable':
            edge_features.append([0, 0, 0, 0, 1])

    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    return edge_features


def edge_att_extractor_v2(G):
    edge_features = []
    for edge in range(len(edge_extract(G))):
        if edge_extract(G)[edge][2]['action'] == 'next':
            edge_features.append([1, 0, 0])
        elif edge_extract(G)[edge][2]['action'] == 'load':
            edge_features.append([0, 1, 0])
        elif edge_extract(G)[edge][2]['action'] == 'unload':
            edge_features.append([0, 0, 1])
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    return edge_features




## Edge_weight matrix using torch_geometric.node2vec one hot encoding for edge feature
def edge_feature_extractor(G):
    ## Combine edge weight and edge features using torch.cat
    edge_feature = torch.cat([edge_weight_extractor(G).unsqueeze(1), edge_att_extractor(G)], dim=1)
    return edge_feature
    
    



### ---------------------------------------------------------------------------



## Node feature matrix for v5
def node_feature_raw_v5(graph, feature='representation'):
    ## do one hot encoding for the node feature using the node name + representation
    node_features = []
    for i in range(len(node_extract(graph))):
        if 'stop' in node_extract(graph)[i][0]:
            node_features.append([1, 0, 0, node_extract(graph)[i][1][feature]])
        elif 'v' in node_extract(graph)[i][0] and 'd' not in node_extract(graph)[i][0]:
            node_features.append([0, 1, 0, node_extract(graph)[i][1][feature]])
        elif 'v' in node_extract(graph)[i][0] and 'd' in node_extract(graph)[i][0]:
            node_features.append([0, 0, 1, node_extract(graph)[i][1][feature]])
    ## convert the node into tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)
    return node_features
    


def node_feature_raw(graph, feature='representation'):
    ## do one hot encoding for the node feature using the node name + representation
    node_features = []
    for i in range(len(node_extract(graph))):
        if 'stop' in node_extract(graph)[i][0]:
            node_features.append([1, 0, 0, node_extract(graph)[i][1][feature]])
        elif 'v' in node_extract(graph)[i][0]:
            node_features.append([0, 1, 0, node_extract(graph)[i][1][feature]])
        elif 'd' in node_extract(graph)[i][0]:
            node_features.append([0, 0, 1, node_extract(graph)[i][1][feature]])
    ## convert the node into tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)
    return node_features


def node_feature_raw_v2(graph, feature='representation'):
    node_features = []
    for i in range(len(node_extract(graph))):
        if 'stop' in node_extract(graph)[i][0]:
            node_features.append([1, 0, node_extract(graph)[i][1][feature]])
        elif 'v' in node_extract(graph)[i][0] and 'd' not in node_extract(graph)[i][0]:
            node_features.append([0, 1, node_extract(graph)[i][1][feature]])
        
    ## Convert the node into tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)
    return node_features


## Node feature matrix using torch_geometric.node2vec, the result should be n-nodes x 64-features
def node_feature_node2vec(graph):
    ## Train Node2Vec model on the combined graph
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    ## Generate the edge_index for the graph
    data = from_networkx(graph)
    ## Learn the node embedding individually
    model = Node2Vec_2(data.edge_index, embedding_dim=32, walk_length=40,
                    context_size=15, walks_per_node=25,
                    num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()  ## Set the model to training mode
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()  
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))  ## Compute the loss for the batch
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item()
        return total_loss / len(loader)
    for epoch in range(1, 71):
        loss = train()
        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    data = from_networkx(graph)

    node_embeddings = model(torch.arange(data.num_nodes)).detach().cpu().numpy()

    ## Convert the node_embeddings into tensor
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float32)

    return node_embeddings



### ---------------------------------------------------------------------------




#%%#

if __name__ == '__main__':
    feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/feasible'

    # infeasible_data_dir_11 = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/infeasible'

    infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/soft'

    feasible_graphs = []
    for file in os.listdir(feasible_data_dir):
        if file.endswith('.json') and 'solution' not in file:
            data = read_json_file(os.path.join(feasible_data_dir, file))
            G = json_to_graph_v2_weight(data)
            feasible_graphs.append(G)


    ## Originally load all 100K infeasible graphs
    infeasible_graphs = []
    for file in os.listdir(infeasible_data_dir):
        if file.endswith('.json') and 'solution' not in file:
            data = read_json_file(os.path.join(infeasible_data_dir, file))
            G = json_to_graph_v2_weight(data)
            infeasible_graphs.append(G)



    ## TODO: IMPORTANT !!
    ## Using extract_graph_features_v2 for v3_3, v3_4, and v2
    ## Using extract_graph_features_v3 for v5

    feasible_features_list = [extract_graph_features_v2(graph) for graph in feasible_graphs]
    infeasible_features_list = [extract_graph_features_v2(graph) for graph in infeasible_graphs]


    ## COnvert the list into dataframe
    feasible_features_df = pd.DataFrame(feasible_features_list)
    infeasible_features_df = pd.DataFrame(infeasible_features_list)


    ## Standardize the features
    scaler = StandardScaler()
    feasible_scaled_features = scaler.fit_transform(feasible_features_df)
    infeasible_scaled_features = scaler.transform(infeasible_features_df)


    ## Select the most similar feasible and infeasible graphs
    selected_infeasible_graphs = []
    remaining_infeasible_features = infeasible_scaled_features.copy()
    remaining_infeasible_graphs = infeasible_graphs.copy()


    for feasible_feature, feasible_graph in zip(feasible_scaled_features, feasible_graphs):
        distances = cdist([feasible_feature], remaining_infeasible_features, 'euclidean').flatten()
        closest_idx = np.argmin(distances)
        
        ## Select and remove the closest infeasible graph
        selected_infeasible_graphs.append(remaining_infeasible_graphs.pop(closest_idx))
        remaining_infeasible_features = np.delete(remaining_infeasible_features, closest_idx, axis=0)


    ## Then select 2610 infeasible graphs from the selected infeasible graphs
    selected_infeasible_graphs = np.random.choice(selected_infeasible_graphs, 25935, replace=False)
    selected_infeasible_graphs = selected_infeasible_graphs.tolist()


    ## Assign the label for the feasible and infeasible graphs
    y1 = torch.tensor([1], dtype=torch.long)
    y0 = torch.tensor([0], dtype=torch.long)


    ## Individual embedding data save location
    save_dir_feasible = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/feasible/raw_1M/v2/'

    ## Google drive save location
    ## save_dir_feasible = '/Users/ttonny0326/Library/CloudStorage/GoogleDrive-shengyic59@gmail.com/My Drive/Dissertation/data/processed/feasible/raw_1M/v3_3/'


    save_dir_infeasible = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/infeasible/raw_1M/v2/'

    ## Google drive save location
    ## save_dir_infeasible = '/Users/ttonny0326/Library/CloudStorage/GoogleDrive-shengyic59@gmail.com/My Drive/Dissertation/data/processed/infeasible/raw_1M/v3_3/'



    # for idx, graph in enumerate(feasible_graphs):
    #     node_features = node_feature_node2vec(graph)
    #     edge_dd = edge_index_extractor(graph)
    #     edge_we = edge_weight_extractor(graph)
    #     edge_att = edge_att_extractor(graph)
    #     edge_feature = torch.cat([edge_we.unsqueeze(1), edge_att], dim=1)
    #     data = Data(x=node_features, edge_index=edge_dd, y=y1, edge_weight=edge_we, edge_attr=edge_att, edge_feature=edge_feature)
    #     save_path = os.path.join(save_dir_feasible, f'feasible_graph_{idx}.pt')
    #     torch.save(data, save_path)
    #     print(f'Saved {save_path}')


    
    ##### Save the selected infeasible graphs -----------------------------------------------------
    with open('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/feasible/raw/v3_3/feasible_graphs.pkl', 'wb') as f:
        pickle.dump(feasible_graphs, f)
    


    with open('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/infeasible/raw/v5/selected_graphs.pkl', 'wb') as f:
        pickle.dump(selected_infeasible_graphs, f)



    ### -----------------------------------------------------------------------------------------
    ## LOAD 


    ## Load the selected infeasible graphs
    with open('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/feasible/n2v/v3_3/feasible_graphs.pkl', 'rb') as f:
        feasible_graphs = pickle.load(f)



    ## Infeasible graphs
    with open('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/infeasible/n2v/v3_3/selected_graphs.pkl', 'rb') as f:
        selected_infeasible_graphs = pickle.load(f)

    

    ##### -----------------------------------------------------------------------------------------

    y1 = torch.tensor([1], dtype=torch.long)
    y0 = torch.tensor([0], dtype=torch.long)

    start_idx = 0
    

    for idx, graph in enumerate(feasible_graphs[start_idx:], start=start_idx):
        ## TODO: v2: node_feature_raw_v2
        ## TODO: v5: node_feature_raw_v5
        node_features = node_feature_raw_v2(graph)
        edge_dd = edge_index_extractor(graph)
        edge_we = edge_weight_extractor(graph)
        ## TODO: v2: eae_v2
        ## TODO: v3: eae_v3
        edge_att = edge_att_extractor_v2(graph)
        edge_feature = torch.cat([edge_we.unsqueeze(1), edge_att], dim=1)

        data = Data(x=node_features, edge_index=edge_dd, y=y1, edge_weight=edge_we, edge_attr=edge_att, edge_feature=edge_feature)
        save_path = os.path.join(save_dir_feasible, f'feasible_graph_{idx}.pt')
        torch.save(data, save_path)
        print(f'Saved {save_path}')


    
    for idx, graph in enumerate(selected_infeasible_graphs[start_idx:], start=start_idx):
        ## TODO: v2: node_feature_raw_v2
        ## TODO: v5: node_feature_raw_v5
        node_features = node_feature_raw_v2(graph)
        edge_dd = edge_index_extractor(graph)
        edge_we = edge_weight_extractor(graph)
        ## TODO: v3: eae_v3
        ## TODO: v2: eae_v2
        edge_att = edge_att_extractor_v2(graph)
        edge_feature = torch.cat([edge_we.unsqueeze(1), edge_att], dim=1)

        data = Data(x=node_features, edge_index=edge_dd, y=y0, edge_weight=edge_we, edge_attr=edge_att, edge_feature=edge_feature)
        save_path = os.path.join(save_dir_infeasible, f'infeasible_graph_{idx}.pt')
        torch.save(data, save_path)
        print(f'Saved {save_path}')


    ## COnvert all y in infeasible data into 0
    # for data in infeasible_data:
    #     data.y = y0



    
    

    ### ---------------------------------------------------------------------------





#%%#
### Step-3-1-1 node2vec_multi-graphs embedding for GraphClassification ###########################
### Train the node2vec model on one graph - 100K instances - node2vec package - separate graphs



## Load the feasible and infeasible graph data from /Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/feasible
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/feasible'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/infeasible'



## Node2vec for GraphClassification 
def graph_to_vector_node2vec(g, agg_method='sum'):
    node2vec = Node2Vec(g, dimensions=64, walk_length=200, num_walks=300, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=10)
    
    ## Get the node vectors
    node_vectors = []
    for node in g.nodes():
        node_vectors.append(model.wv[str(node)])

    ## COnvert the node vectors into graph vector
    if agg_method == 'mean':
        graph_vector = np.mean(node_vectors, axis=0)
    elif agg_method == 'sum':
        graph_vector = np.sum(node_vectors, axis=0)
    elif agg_method == 'max':
        graph_vector = np.max(node_vectors, axis=0)
    
    return graph_vector





if __name__ == '__main__':
    feasible_graphs = []
    for file in os.listdir(feasible_data_dir):
        ## Select the file with json format and file name do not contain solution
        if file.endswith('.json') and 'solution' not in file:
            data = read_json_file(os.path.join(feasible_data_dir, file))
            G = json_to_graph_v3_2(data)
            feasible_graphs.append(G)


    infeasible_graphs = []
    ## Select the file with json format
    infeasible_files = [file for file in os.listdir(infeasible_data_dir) if file.endswith('.json')]
    ## Random select 2610 files from those files
    infeasible_files = np.random.choice(infeasible_files, 2610, replace=False)
    for file in infeasible_files:
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        G = json_to_graph_v3_2(data)
        infeasible_graphs.append(G)


    ## Transform the graph vector as data frame format
    feasible_graph_vectors = []
    for graph in feasible_graphs:
        # print(f"Processing feasible graph {i+1}/{len(feasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, agg_method='mean')
        feasible_graph_vectors.append(graph_vector)


    infeasible_graph_vectors = []
    for graph in infeasible_graphs:
        # print(f"Processing infeasible graph {k+1}/{len(infeasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, agg_method='mean')
        infeasible_graph_vectors.append(graph_vector)

   
    ## Store the graph vectors into data frame
    feasible_df = pd.DataFrame(feasible_graph_vectors)
    feasible_df['label'] = 1
    infeasible_df = pd.DataFrame(infeasible_graph_vectors)
    infeasible_df['label'] = 0

    ## Combine the feasible and infeasible graph vectors
    graph_df = pd.concat([feasible_df, infeasible_df], axis=0)

    ## Save the graph vectors into csv file
    # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/graph_vectors_node2vec.csv', index=False)



### ---------------------------------------------------------------------------



#%%#
### Step-3-1-2 node2vec_one graph embedding for NodeClassification ################################
### Train the node2vec model on one graph - 100K instances - node2vec package - combined graph

# ## Load the feasible and infeasible graph data
# feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/feasible'

# infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/infeasible'


# ## Load the feasible graph data from the data directory 
# feasible_graphs = []
# for file in os.listdir(feasible_data_dir):
#     if file.endswith('.json'):
#         data = read_json_file(os.path.join(feasible_data_dir, file))
#         G = json_to_graph_v4(data)
#         feasible_graphs.append(G)


# infeasible_graphs = []
# ##  randomly select 1000 infeasible instances from the data directory to make the dataset balanced
# infeasible_files = np.random.choice(os.listdir(infeasible_data_dir), 303, replace=False)
# for file in infeasible_files:
#     if file.endswith('.json'):
#         data = read_json_file(os.path.join(infeasible_data_dir, file))
#         G = json_to_graph_v4(data)
#         infeasible_graphs.append(G)


## Load the feasible and infeasible graph data
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/feasible'


infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/infeasible'



## Function to generate graph-level embeddings using the trained Node2Vec model
def graph_to_vector_node2vec(g, model, agg_method='sum'):
    node_vectors = []
    for node in g.nodes():
        node_vectors.append(model.wv[str(node)])

    if agg_method == 'mean':
        graph_vector = np.mean(node_vectors, axis=0)
    elif agg_method == 'sum':
        graph_vector = np.sum(node_vectors, axis=0)
    elif agg_method == 'max':
        graph_vector = np.max(node_vectors, axis=0)
    
    return graph_vector



if __name__ == '__main__':
    feasible_graphs = []
    for file in os.listdir(feasible_data_dir):
        ## Select the file with json format and file name do not contain solution
        if file.endswith('.json') and 'solution' not in file:
            data = read_json_file(os.path.join(feasible_data_dir, file))
            G = json_to_graph_v3_2(data)
            feasible_graphs.append(G)


    infeasible_graphs = []
    ## Select the file with json format
    infeasible_files = [file for file in os.listdir(infeasible_data_dir) if file.endswith('.json')]
    ## Random select 2610 files from those files
    infeasible_files = np.random.choice(infeasible_files, 2610, replace=False)
    for file in infeasible_files:
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        G = json_to_graph_v3_2(data)
        infeasible_graphs.append(G)


    ## Combine all graphs into one big graph for Node2Vec training
    combined_graph = nx.MultiDiGraph()
    for graph in feasible_graphs + infeasible_graphs:
        combined_graph = nx.compose(combined_graph, graph)


    ## Train Node2Vec model on the combined graph
    node2vec = Node2Vec(combined_graph, dimensions=64, walk_length=200, num_walks=300, workers=10)
    model = node2vec.fit(window=10, min_count=1, batch_words=10)


    ## Transform the graph vector as data frame format
    feasible_graph_vectors = []
    for i, graph in enumerate(feasible_graphs):
        print(f"Processing feasible graph {i+1}/{len(feasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, model, agg_method='mean')
        feasible_graph_vectors.append(graph_vector)


    infeasible_graph_vectors = []
    for k, graph in enumerate(infeasible_graphs):
        print(f"Processing infeasible graph {k+1}/{len(infeasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, model, agg_method='mean')
        infeasible_graph_vectors.append(graph_vector)


    ## Store the graph vectors into data frame
    feasible_df = pd.DataFrame(feasible_graph_vectors)
    feasible_df['label'] = 1
    infeasible_df = pd.DataFrame(infeasible_graph_vectors)
    infeasible_df['label'] = 0

    ## Combine the feasible and infeasible graph vectors
    graph_df = pd.concat([feasible_df, infeasible_df], axis=0)

    ## Save the graph vectors into csv file
    # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v4_graph_vectors_node2vec_mean_one.csv', index=False)



#%%#
### Step-3-1-3 node2vec_ torch gem embedding for NodeClassification ############################

## Load the feasible and infeasible graph data
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/feasible'


infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/infeasible'


def graph_to_vector_node2vec(graph, model, agg_method='mean'):
    data = from_networkx(graph)
    node_embeddings = model(torch.arange(data.num_nodes)).detach().cpu().numpy()
    if agg_method == 'mean':
        graph_vector = np.mean(node_embeddings, axis=0)
    elif agg_method == 'sum':
        graph_vector = np.sum(node_embeddings, axis=0)
    elif agg_method == 'max':
        graph_vector = np.max(node_embeddings, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")
    return graph_vector




if __name__ == '__main__':
    feasible_graphs = []
    for file in os.listdir(feasible_data_dir):
        ## Select the file with json format and file name do not contain solution
        if file.endswith('.json') and 'solution' not in file:
            data = read_json_file(os.path.join(feasible_data_dir, file))
            G = json_to_graph_v3_3(data)
            feasible_graphs.append(G)


    infeasible_graphs = []
    ## Select the file with json format
    infeasible_files = [file for file in os.listdir(infeasible_data_dir) if file.endswith('.json')]
    ## Random select 2610 files from those files
    infeasible_files = np.random.choice(infeasible_files, 2610, replace=False)
    for file in infeasible_files:
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        G = json_to_graph_v3_3(data)
        infeasible_graphs.append(G)


    ## Combine all graphs into one big graph for Node2Vec training
    combined_graph = nx.MultiDiGraph()
    for graph in feasible_graphs + infeasible_graphs:
        combined_graph = nx.compose(combined_graph, graph)


    ## Train Node2Vec model on the combined graph
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    
    ## Generate the edge_index for the graph
    data = from_networkx(combined_graph)
    model = Node2Vec_2(data.edge_index, embedding_dim=64, walk_length=200,
                    context_size=20, walks_per_node=300,
                    num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()  # Set the model to training mode
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()  # Set the gradients to zero
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # Compute the loss for the batch
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Optimize the parameters
            total_loss += loss.item()
        return total_loss / len(loader)
    
    for epoch in range(1, 101):
        loss = train()
        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


    ## Transform the graph vector as data frame format
    feasible_graph_vectors = []
    for i, graph in enumerate(feasible_graphs):
        print(f"Processing feasible graph {i+1}/{len(feasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, model, agg_method='sum')
        feasible_graph_vectors.append(graph_vector)


    infeasible_graph_vectors = []
    for k, graph in enumerate(infeasible_graphs):
        print(f"Processing infeasible graph {k+1}/{len(infeasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, model, agg_method='sum')
        infeasible_graph_vectors.append(graph_vector)


    ## Store the graph vectors into data frame
    feasible_df = pd.DataFrame(feasible_graph_vectors)
    feasible_df['label'] = 1
    infeasible_df = pd.DataFrame(infeasible_graph_vectors)
    infeasible_df['label'] = 0
    
    ## Combine the feasible and infeasible graph vectors
    graph_df = pd.concat([feasible_df, infeasible_df], axis=0)

    ## Save the graph vectors into csv file
    # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v6_graph_vectors_node2vec_2_sum_one.csv', index=False)


## Save the node2vec model
save_path = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/models/node2vec_v6.pt'
# torch.save(model.state_dict(), save_path)

## Load the node2vec model
## Initialize a new model instance with the same parameters
# loaded_model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=80,
#                         context_size=10, walks_per_node=80,
#                         num_negative_samples=1, p=1, q=1, sparse=True)

# # Load the state dictionary into the model
# loaded_model.load_state_dict(torch.load(save_path))

# # Ensure the model is in evaluation mode
# loaded_model.eval()






#%%#
### Step-3-2-structure2vec for GraphClassification #################################################
## Second embedding approach is to use the structure2vec algorithm to convert the graph data into vector data


## Define Structure2vec algorithm
class Structure2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_iterations):
        super(Structure2Vec, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W3 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))
    
    def linear_transform(self, x, W, b):
        return torch.matmul(x, W) + b
    
    def forward(self, features, adj):
        h = self.linear_transform(features, self.W1, self.b1)
        
        for _ in range(self.num_iterations):
            m = torch.matmul(adj, h)
            h = F.relu(self.linear_transform(m + h, self.W2, self.b2))
        
        return h  ## Return node embeddings


## Feature Extractor - one feature per node
def extract_node_features(graph):
    return np.array([graph.nodes[node]['features'] for node in graph.nodes()])


## Feature Extractor - pad features to a fixed length
def extract_node_features_v2(graph, input_dim):
    features = []
    for node in graph.nodes():
        node_feature = graph.nodes[node].get('features', [])
        if len(node_feature) < input_dim:
            node_feature = node_feature + [0] * (input_dim - len(node_feature))
        else:
            node_feature = node_feature[:input_dim]
        features.append(node_feature)
    
    return np.array(features)


## Generate graph embeddings using the structure2vec algorithm (graph level)
def generate_graph_embedding(model, graph, features, agg_method='mean'):
    ## Ensure adjacency matrix is in the correct format
    adj = nx.adjacency_matrix(graph).todense()
    adj = torch.tensor(adj, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)
    
    ## Generate node embeddings using the model
    node_embeddings = model(features, adj)
    
    ## Debug: Print shape of node embeddings
    print("Node embeddings shape:", node_embeddings.shape)
    
    ## Aggregate node embeddings into a single graph-level embedding
    if agg_method == 'mean':
        graph_embedding = torch.mean(node_embeddings, dim=0)
    elif agg_method == 'sum':
        graph_embedding = torch.sum(node_embeddings, dim=0)
    elif agg_method == 'max':
        graph_embedding = torch.max(node_embeddings, dim=0).values
    
    
    return graph_embedding.detach().numpy()


## Model parameters
input_dim = 20
hidden_dim = 64
num_iterations = 10


## Initial model
model = Structure2Vec(input_dim, hidden_dim, num_iterations)


if __name__ == '__main__':
    ## Generate graph embeddings
    feasible_graph_vectors = []
    for i in feasible_graphs:
        ## Extract the node feature form the graph, instead of random feature
        features = extract_node_features(i)
        graph_embedding = generate_graph_embedding(model, i, features, agg_method='sum')
        feasible_graph_vectors.append(graph_embedding)

    infeasible_graph_vectors = []
    for k in infeasible_graphs:
        features = extract_node_features(k)
        graph_embedding = generate_graph_embedding(model, k, features, agg_method='sum')
        infeasible_graph_vectors.append(graph_embedding)


    ## Store the graph vectors into data frame
    feasible_df = pd.DataFrame(feasible_graph_vectors)
    feasible_df['label'] = 1
    infeasible_df = pd.DataFrame(infeasible_graph_vectors)
    infeasible_df['label'] = 0

    ## Combine the feasible and infeasible graph vectors
    graph_df = pd.concat([feasible_df, infeasible_df], axis=0)

    ## Save the graph vectors into csv file
    # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/graph_vectors_structure2vec.csv', index=False)



### ---------------------------------------------------------------------------


#%%#
##Test for s2v

# Load the feasible and infeasible graph data
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/feasible'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/infeasible'

# # Load feasible graphs
# feasible_graphs = []
# for file in os.listdir(feasible_data_dir):
#     if file.endswith('.json'):
#         data = read_json_file(os.path.join(feasible_data_dir, file))
#         G = json_to_graph_v4(data)
#         feasible_graphs.append(G)

# # Load infeasible graphs
# infeasible_graphs = []
# infeasible_files = np.random.choice(os.listdir(infeasible_data_dir), 303, replace=False)
# for file in infeasible_files:
#     if file.endswith('.json'):
#         data = read_json_file(os.path.join(infeasible_data_dir, file))
#         G = json_to_graph_v4(data)
#         infeasible_graphs.append(G)


# class Structure2Vec(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_iterations):
#         super(Structure2Vec, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_iterations = num_iterations
#         self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
#         self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         self.W3 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         self.b1 = nn.Parameter(torch.zeros(hidden_dim))
#         self.b2 = nn.Parameter(torch.zeros(hidden_dim))
    
#     def linear_transform(self, x, W, b):
#         return torch.matmul(x, W) + b
    
#     def forward(self, features, adj):
#         h = self.linear_transform(features, self.W1, self.b1)
        
#         for _ in range(self.num_iterations):
#             m = torch.matmul(adj, h)
#             h = F.relu(self.linear_transform(m + h, self.W2, self.b2))
        
#         g = torch.mean(h, dim=0)  # Graph-level embedding by mean pooling
#         g = self.linear_transform(g, self.W3, self.b1)
        
#         return g  # Return the graph-level embedding


# def generate_graph_embedding_s2v(model, graph, features):
#     adj = nx.adjacency_matrix(graph).todense()
#     adj = torch.tensor(adj, dtype=torch.float32)
#     features = torch.tensor(features, dtype=torch.float32)
    
#     graph_embedding = model(features, adj)
    
#     return graph_embedding.detach().numpy()


# # Model parameters
# input_dim = 10  # Number of input features
# hidden_dim = 64
# num_iterations = 10

# # Initialize the Structure2Vec model
# s2v_model = Structure2Vec(input_dim, hidden_dim, num_iterations)


# # Example function to extract node features from a graph
# def extract_node_features(graph, input_dim):
#     features = []
#     for node in graph.nodes():
#         node_feature = graph.nodes[node].get('features', [])
#         if len(node_feature) < input_dim:
#             node_feature = node_feature + [0] * (input_dim - len(node_feature))
#         else:
#             node_feature = node_feature[:input_dim]
#         features.append(node_feature)
#     return np.array(features)


# # Prepare training data for Structure2Vec
# X_graphs = feasible_graphs + infeasible_graphs
# y_labels = [1] * len(feasible_graphs) + [0] * len(infeasible_graphs)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_graphs, y_labels, test_size=0.2, random_state=42)

# train_data = [(extract_node_features(graph, input_dim), nx.adjacency_matrix(graph).todense(), label)
#               for graph, label in zip(X_train, y_train)]

# test_data = [(extract_node_features(graph, input_dim), nx.adjacency_matrix(graph).todense(), label)
#               for graph, label in zip(X_test, y_test)]


# def train_s2v(model, data, epochs=100, learning_rate=0.01):
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()  # Using MSELoss for regression task (if embeddings are target)

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()

#         losses = []
#         for features, adj, label in data:
#             features = torch.tensor(features, dtype=torch.float32)
#             adj = torch.tensor(adj, dtype=torch.float32)
#             label = torch.tensor(label, dtype=torch.float32)  # Ensure correct data type

#             output = model(features, adj)
#             loss = loss_fn(output, label)
#             losses.append(loss.item())

#             loss.backward()
#             optimizer.step()
        
#         if epoch % 10 == 0:
#             avg_loss = np.mean(losses)
#             print(f'Epoch {epoch}, Loss: {avg_loss}')

# train_s2v(s2v_model, train_data)

# if __name__ == '__main__':
#     feasible_graph_vectors = []
#     for i in feasible_graphs:
#         features = extract_node_features(i, input_dim)
#         graph_embedding = generate_graph_embedding_s2v(s2v_model, i, features)
#         feasible_graph_vectors.append(graph_embedding)

#     infeasible_graph_vectors = []
#     for k in infeasible_graphs:
#         features = extract_node_features(k, input_dim)
#         graph_embedding = generate_graph_embedding_s2v(s2v_model, k, features)
#         infeasible_graph_vectors.append(graph_embedding)


#     feasible_df = pd.DataFrame(feasible_graph_vectors)
#     feasible_df['label'] = 1
#     infeasible_df = pd.DataFrame(infeasible_graph_vectors)
#     infeasible_df['label'] = 0

#     graph_df = pd.concat([feasible_df, infeasible_df], axis=0)

#     # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/graph_vectors_structure2vec.csv', index=False)













#%%#
### step-3-3-transformer for GraphClassification #################################################
## Third embedding approach is to use the GTN(transformer based algorithm) to convert the graph data into vector data


## Define the GTN model for the graph embedding 
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(GraphTransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
    
    def forward(self, src, src_mask=None):
        src2 = self.attention(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, hidden_dim))
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        x = self.embedding(x) + self.positional_encoding
        for layer in self.layers:
            x = layer(x)
        return x  # Return node embeddings without aggregation


## Feature Extractor
def extract_node_features(graph):
    return np.array([graph.nodes[node]['features'] for node in graph.nodes()])


## Graph Embedding Generator (graph level)
def generate_graph_embedding(model, graph, features, agg_method='sum'):
    adj = nx.adjacency_matrix(graph).todense()
    adj = torch.tensor(adj, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)
    
    node_embeddings = model(features, adj)
    
    print("Node embeddings shape:", node_embeddings.shape)
    
    if agg_method == 'mean':
        graph_embedding = torch.mean(node_embeddings, dim=0)
    elif agg_method == 'sum':
        graph_embedding = torch.sum(node_embeddings, dim=0)
    elif agg_method == 'max':
        graph_embedding = torch.max(node_embeddings, dim=0).values
    
    print("Graph embedding shape after aggregation ({}): {}".format(agg_method, graph_embedding.shape))
    
    return graph_embedding.detach().numpy()



input_dim = 10
hidden_dim = 64
num_heads = 8
num_layers = 3

model = GraphTransformer(input_dim, hidden_dim, num_heads, num_layers)



if __name__ == '__main__':
    feasible_graph_vectors = []
    for i in feasible_graphs:
        features = extract_node_features(i)
        graph_embedding = generate_graph_embedding(model, i, features, agg_method='sum')
        feasible_graph_vectors.append(graph_embedding)

    infeasible_graph_vectors = []
    for k in infeasible_graphs:
        features = extract_node_features(k)
        graph_embedding = generate_graph_embedding(model, k, features, agg_method='sum')
        infeasible_graph_vectors.append(graph_embedding)


    feasible_df = pd.DataFrame(feasible_graph_vectors)
    feasible_df['label'] = 1
    infeasible_df = pd.DataFrame(infeasible_graph_vectors)
    infeasible_df['label'] = 0

    graph_df = pd.concat([feasible_df, infeasible_df], axis=0)

    # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/graph_vectors_transformer.csv', index=False)



### ---------------------------------------------------------------------------



#%%#
### Step-4-1-graph2vec for NodeClassification #######################################################
## Fourth embedding approach is to use the graph2vec algorithm to convert the graph data into vector data




