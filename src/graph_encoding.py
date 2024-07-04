#%%#
### Step-1 ######################################################################
### Read the json file

import os
import json
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from karateclub import FeatherGraph
from sklearn.model_selection import train_test_split
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
### Step-3-1-1 node2vec_multi-graphs embedding for GraphClassification ###########################
### Train the node2vec model on multiple graphs
### Convert the graph data into vector data - for classify the graph is feasible or not


## Load the feasible and infeasible graph data from /Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/feasible
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/feasible'


infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/instances/infeasible'


## Load the feasible graph data from the data directory 
feasible_graphs = []
for file in os.listdir(feasible_data_dir):
    if file.endswith('.json'):
        data = read_json_file(os.path.join(feasible_data_dir, file))
        G = json_to_graph_v4(data)
        feasible_graphs.append(G)


infeasible_graphs = []
##  randomly select 1000 infeasible instances from the data directory to make the dataset balanced
infeasible_files = np.random.choice(os.listdir(infeasible_data_dir), 303, replace=False)
for file in infeasible_files:
    if file.endswith('.json'):
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        G = json_to_graph_v4(data)
        infeasible_graphs.append(G)



## Node2vec for GraphClassification 
def graph_to_vector_node2vec(g, agg_method='sum'):
    node2vec = Node2Vec(g, dimensions=64, walk_length=30, num_walks=100, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    ## Get the node vectors
    node_vectors = []
    for node in g.nodes():
        node_vectors.append(model.wv[node])

    ## COnvert the node vectors into graph vector
    if agg_method == 'mean':
        graph_vector = np.mean(node_vectors, axis=0)
    elif agg_method == 'sum':
        graph_vector = np.sum(node_vectors, axis=0)
    elif agg_method == 'max':
        graph_vector = np.max(node_vectors, axis=0)
    
    return graph_vector



if __name__ == '__main__':
    ## Transform the graph vector as data frame format
    feasible_graph_vectors = []
    for i, graph in enumerate(feasible_graphs):
        print(f"Processing feasible graph {i+1}/{len(feasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, agg_method='sum')
        feasible_graph_vectors.append(graph_vector)

    infeasible_graph_vectors = []
    for k, graph in enumerate(infeasible_graphs):
        print(f"Processing infeasible graph {k+1}/{len(infeasible_graphs)}")
        graph_vector = graph_to_vector_node2vec(graph, agg_method='sum')
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
### Train the node2vec model on one graph - 100K instances

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


feasible_graphs = []
for file in os.listdir(feasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json') and 'solution' not in file:
        data = read_json_file(os.path.join(feasible_data_dir, file))
        G = json_to_graph_v4(data)
        feasible_graphs.append(G)


infeasible_graphs = []
## Select the file with json format
infeasible_files = [file for file in os.listdir(infeasible_data_dir) if file.endswith('.json')]
## Random select 2610 files from those files
infeasible_files = np.random.choice(infeasible_files, 2610, replace=False)
for file in infeasible_files:
    data = read_json_file(os.path.join(infeasible_data_dir, file))
    G = json_to_graph_v4(data)
    infeasible_graphs.append(G)



## Combine all graphs into one big graph for Node2Vec training
combined_graph = nx.MultiDiGraph()
for graph in feasible_graphs + infeasible_graphs:
    combined_graph = nx.compose(combined_graph, graph)


## Train Node2Vec model on the combined graph
node2vec = Node2Vec(combined_graph, dimensions=64, walk_length=80, num_walks=80, workers=10)
model = node2vec.fit(window=10, min_count=1, batch_words=10)


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
    # graph_df.to_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v4_graph_vectors_node2vec_mean_one.csv', index=False)




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

# Load feasible graphs
feasible_graphs = []
for file in os.listdir(feasible_data_dir):
    if file.endswith('.json'):
        data = read_json_file(os.path.join(feasible_data_dir, file))
        G = json_to_graph_v4(data)
        feasible_graphs.append(G)

# Load infeasible graphs
infeasible_graphs = []
infeasible_files = np.random.choice(os.listdir(infeasible_data_dir), 303, replace=False)
for file in infeasible_files:
    if file.endswith('.json'):
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        G = json_to_graph_v4(data)
        infeasible_graphs.append(G)


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
        
        g = torch.mean(h, dim=0)  # Graph-level embedding by mean pooling
        g = self.linear_transform(g, self.W3, self.b1)
        
        return g  # Return the graph-level embedding


def generate_graph_embedding_s2v(model, graph, features):
    adj = nx.adjacency_matrix(graph).todense()
    adj = torch.tensor(adj, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)
    
    graph_embedding = model(features, adj)
    
    return graph_embedding.detach().numpy()


# Model parameters
input_dim = 10  # Number of input features
hidden_dim = 64
num_iterations = 10

# Initialize the Structure2Vec model
s2v_model = Structure2Vec(input_dim, hidden_dim, num_iterations)


# Example function to extract node features from a graph
def extract_node_features(graph, input_dim):
    features = []
    for node in graph.nodes():
        node_feature = graph.nodes[node].get('features', [])
        if len(node_feature) < input_dim:
            node_feature = node_feature + [0] * (input_dim - len(node_feature))
        else:
            node_feature = node_feature[:input_dim]
        features.append(node_feature)
    return np.array(features)


# Prepare training data for Structure2Vec
X_graphs = feasible_graphs + infeasible_graphs
y_labels = [1] * len(feasible_graphs) + [0] * len(infeasible_graphs)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_graphs, y_labels, test_size=0.2, random_state=42)

train_data = [(extract_node_features(graph, input_dim), nx.adjacency_matrix(graph).todense(), label)
              for graph, label in zip(X_train, y_train)]

test_data = [(extract_node_features(graph, input_dim), nx.adjacency_matrix(graph).todense(), label)
              for graph, label in zip(X_test, y_test)]


def train_s2v(model, data, epochs=100, learning_rate=0.01):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # Using MSELoss for regression task (if embeddings are target)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = []
        for features, adj, label in data:
            features = torch.tensor(features, dtype=torch.float32)
            adj = torch.tensor(adj, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)  # Ensure correct data type

            output = model(features, adj)
            loss = loss_fn(output, label)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            avg_loss = np.mean(losses)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

train_s2v(s2v_model, train_data)

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




