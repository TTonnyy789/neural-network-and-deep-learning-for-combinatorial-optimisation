#%%#
### Step-1 ######################################################################
### Read the json file

import os
import json
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from data_preprocessing import *
from graph_encoding import *



#%%#
### Step-2 ######################################################################
### Load the 100K instance dataset

## Define the latest data directory
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/feasible'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/infeasible'


## Convert them into graph format
feasible_graphs = []
for file in os.listdir(feasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json') and 'solution' not in file:
        data = read_json_file(os.path.join(feasible_data_dir, file))
        G = json_to_graph_v5(data)
        feasible_graphs.append(G)

## Raw json data
feasible_graphs_v2 = []
for file in os.listdir(feasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json') and 'solution' not in file:
        data = read_json_file(os.path.join(feasible_data_dir, file))
        # G = json_to_graph_v5(data)
        feasible_graphs_v2.append(data)


## Random select 2610 files from those files
infeasible_graphs = []
## Select the file with json format
infeasible_files = [file for file in os.listdir(infeasible_data_dir) if file.endswith('.json')]
## Random select 2610 files from those files
infeasible_files = np.random.choice(infeasible_files, 2610, replace=False)
for file in infeasible_files:
    data = read_json_file(os.path.join(infeasible_data_dir, file))
    G = json_to_graph_v5(data)
    infeasible_graphs.append(G)




infeasible_graphs = []
for file in os.listdir(infeasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json'):
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        G = json_to_graph_v5(data)
        infeasible_graphs.append(G)

## Raw json data
infeasible_graphs_v2 = []
for file in os.listdir(infeasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json'):
        data = read_json_file(os.path.join(infeasible_data_dir, file))
        # G = json_to_graph_v5(data)
        infeasible_graphs_v2.append(data)


### -------------------------------------------------


#%%#
### Step-2 ######################################################################
## Selecting similar graphs

feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/feasible'

infeasible_data_dir_2 = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/infeasible'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/1M_instances/soft'

## Raw json data
## FIXME: use this for 'time' calculate
feasible_graphs_v2 = []
for file in os.listdir(feasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json') and 'solution' not in file:
        data = read_json_file(os.path.join(feasible_data_dir, file))
        # G = json_to_graph_v5(data)
        feasible_graphs_v2.append(data)

## Raw json data
# infeasible_graphs_v2 = []
# for file in os.listdir(infeasible_data_dir):
#     ## Select the file with json format and file name do not contain solution
#     if file.endswith('.json'):
#         data = read_json_file(os.path.join(infeasible_data_dir, file))
#         # G = json_to_graph_v5(data)
#         infeasible_graphs_v2.append(data)


## Raw json data
infeasible_graphs_v2 = []
for file in os.listdir(infeasible_data_dir_2):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json'):
        data = read_json_file(os.path.join(infeasible_data_dir_2, file))
        # G = json_to_graph_v5(data)
        infeasible_graphs_v2.append(data)


# Extract features
feasible_features_list = [extract_json_features(graph) for graph in feasible_graphs_v2]
infeasible_features_list = [extract_json_features(graph) for graph in infeasible_graphs_v2]

# Convert lists to DataFrames
feasible_features_df = pd.DataFrame(feasible_features_list)
infeasible_features_df = pd.DataFrame(infeasible_features_list)

# Standardize features
scaler = StandardScaler()
feasible_scaled_features = scaler.fit_transform(feasible_features_df)
infeasible_scaled_features = scaler.transform(infeasible_features_df)

# Select the most similar infeasible graphs to each feasible graph
selected_infeasible_graphs = []
remaining_infeasible_features = infeasible_scaled_features.copy()
remaining_infeasible_graphs = infeasible_graphs_v2.copy()

for feasible_feature in feasible_scaled_features:
    distances = cdist([feasible_feature], remaining_infeasible_features, 'euclidean').flatten()
    closest_idx = np.argmin(distances)
    
    # Select and remove the closest infeasible graph
    selected_infeasible_graphs.append(remaining_infeasible_graphs.pop(closest_idx))
    remaining_infeasible_features = np.delete(remaining_infeasible_features, closest_idx, axis=0)

# Select 2610 infeasible graphs from the selected infeasible graphs
selected_infeasible_graphs = np.random.choice(selected_infeasible_graphs, 25935, replace=False)
selected_infeasible_graphs.tolist()



# Convert feasible graphs to NetworkX format
feasible_graphs = []
for i in range(len(feasible_graphs_v2)):
    data = json_to_graph_v5_weight(feasible_graphs_v2[i])
    feasible_graphs.append(data)




# infeasible_graphs = []
# for i in range(len(selected_infeasible_graphs)):
#     data = json_to_graph_v5_weight(selected_infeasible_graphs[i])
#     infeasible_graphs.append(data)


infeasible_graphs = []
## Random select 25935 files from those files
random_files = np.random.choice(infeasible_graphs_v2, 25935, replace=False)
for i in range(len(random_files)):
    data = json_to_graph_v5_weight(random_files[i])
    infeasible_graphs.append(data)





#%%#
### Step-2-1 ######################################################################
## Select those infeasible graphs with same number of vehicle nodes as feasible graphs


## Define the latest data directory
feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/feasible'


infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/infeasible'


## Convert them into graph format
# feasible_graphs = []
# for file in os.listdir(feasible_data_dir):
#     ## Select the file with json format and file name do not contain solution
#     if file.endswith('.json') and 'solution' not in file:
#         data = read_json_file(os.path.join(feasible_data_dir, file))
#         G = json_to_graph_v4(data)
#         if len([node for node in G.nodes() if 'v' in node and 'd' not in node]) == 6:
#             feasible_graphs.append(G)


# infeasible_graphs = []
# for file in os.listdir(infeasible_data_dir):
#     ## Select the file with json format and file name do not contain solution
#     if file.endswith('.json'):
#         data = read_json_file(os.path.join(infeasible_data_dir, file))
#         G = json_to_graph_v4(data)
#         ## Select those infeasible graphs that the number of stop is 6 and 7
#         if len([node for node in G.nodes() if 'v' in node and 'd' not in node]) == 6:
#             infeasible_graphs.append(G)


feasible_graphs = []
for file in os.listdir(feasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json') and 'solution' not in file:
        data = read_json_file(os.path.join(feasible_data_dir, file))
        G = json_to_graph_v5(data)
        

## Select the infeasible graphs with the similar feature as feasible graphs such as the number of stops, vehicles, and edges using spicy package







## Measure the whether vehicle unload and load in single stop -> make the solution infeasible
## For infeasible solution, in each stop, the number of loaded vehicles is greater than feasible one


## TODO: Check the single stop's loaded vehicle number

### -------------------------------------------------






#%%#
### Step-3 ######################################################################
### Visualize the distribution of the number of nodes and edges across the dataset


## Feasible graphs' number of nodes, edges, stop nodes, and vehicle nodes
feasible_number_of_nodes = []
for i in feasible_graphs:
    length = len(node_extract(i))
    feasible_number_of_nodes.append(length)

feasible_number_of_edges = []
for k in feasible_graphs:
    length = len(edge_extract(k))
    feasible_number_of_edges.append(length)

feasible_stop_nodes = []
feasible_vehicle_nodes = []

for a in feasible_graphs:
    stop_nodes = [node for node in a.nodes() if 'stop' in node]
    vehicle_nodes = [node for node in a.nodes() if 'v' in node and 'd' not in node]
    feasible_stop_nodes.append(len(stop_nodes))
    feasible_vehicle_nodes.append(len(vehicle_nodes))



### -------------------------------------------------
feasible_unloading_edges = []

## In feasible graphs, select the edge between stop nodes and vehicle nodes with unload
for graph in feasible_graphs:
    ## Calculate the number of edges that 'action' is unload
    le = []
    ed = edge_extract(graph)
    for k in ed:
        if k[2]['action'] == 'unload':
            le.append(len(k))
    feasible_unloading_edges.append(len(le))



### -------------------------------------------------
average_feasible_stop_load_number = []
standard_deviation_feasible_stop_load_number = []

for graph in feasible_graphs:
    stop_load_car_length = []
    bb = node_extract(graph)
    ## If stop is loading v1, v2, then the length is 2 and append it to the list in the first element
    for k in bb:
        ## Select those stop nodes has 'load' key
        if 'load' in k[1]:
            stop_load_car_length.append(len(k[1]['load']))
    standard_deviation_feasible_stop_load_number.append(np.std(stop_load_car_length))
    average_feasible_stop_load_number.append(np.mean(stop_load_car_length))
        

### -------------------------------------------------
## TODO: The index of each vehicle in the solution










### -------------------------------------------------
## Calculate the average time for each vehicle on the deck, to do so, using the 'unload' point minus 'load' point for each vehicle


feasible_avg_vehicle_distance = []
for data in feasible_graphs_v2:
    bb = []
    number_of_vehilce = calculate_vehicle_distances(data).shape[0]
    for i in range(number_of_vehilce):
        bb.append(calculate_vehicle_distances(data).iloc[i, 1])
    feasible_avg_vehicle_distance.append(np.mean(bb))
    

feasible_std_vehicle_distance = []
for data in feasible_graphs_v2:
    hh = []
    number_of_vehilce = calculate_vehicle_distances(data).shape[0]
    for i in range(number_of_vehilce):
        hh.append(calculate_vehicle_distances(data).iloc[i, 1])
    feasible_std_vehicle_distance.append(np.std(hh))






### ---------------------------------------------------------------------------------------------
## Infeasible graphs' number of nodes, edges, stop nodes, and vehicle nodes
infeasible_number_of_nodes = []
for j in infeasible_graphs:
    length = len(node_extract(j))
    infeasible_number_of_nodes.append(length)

infeasible_number_of_edges = []
for m in infeasible_graphs:
    length = len(edge_extract(m))
    infeasible_number_of_edges.append(length)


### -------------------------------------------------
infeasible_stop_nodes = []
infeasible_vehicle_nodes = []

for b in infeasible_graphs:
    stop_nodes = [node for node in b.nodes() if 'stop' in node]
    vehicle_nodes = [node for node in b.nodes() if 'v' in node and 'd' not in node]
    infeasible_stop_nodes.append(len(stop_nodes))
    infeasible_vehicle_nodes.append(len(vehicle_nodes))



### -------------------------------------------------
infeasible_unloading_edges = []

## In infeasible graphs, select the edge between stop nodes and vehicle nodes with unload
for graph in infeasible_graphs:
    ## Calculate the number of edges that 'action' is unload
    le = []
    ed = edge_extract(graph)
    for k in ed:
        if k[2]['action'] == 'unload':
            le.append(len(k))
    infeasible_unloading_edges.append(len(le))



### -------------------------------------------------
average_infeasible_stop_load_number = []
standard_deviation_infeasible_stop_load_number = []

for graph in infeasible_graphs:
    stop_load_car_length = []
    aa = node_extract(graph)
    ## If stop is loading v1, v2, then the length is 2 and append it to the list in the first element
    for o in aa:
        ## Select those stop nodes has 'load' key
        if 'load' in o[1]:
            stop_load_car_length.append(len(o[1]['load']))
    standard_deviation_infeasible_stop_load_number.append(np.std(stop_load_car_length))
    average_infeasible_stop_load_number.append(np.mean(stop_load_car_length))


### -------------------------------------------------
## Calculate the average time for each vehicle on the deck, to do so, using the 'unload' point minus 'load' point for each vehicle

infeasible_avg_vehicle_distance = []
for data in selected_infeasible_graphs:
    cc = []
    number_of_vehilce = calculate_vehicle_distances(data).shape[0]
    for i in range(number_of_vehilce):
        cc.append(calculate_vehicle_distances(data).iloc[i, 1])
    infeasible_avg_vehicle_distance.append(np.mean(cc))


infeasible_std_vehicle_distance = []
for data in selected_infeasible_graphs:
    qq = []
    number_of_vehilce = calculate_vehicle_distances(data).shape[0]
    for i in range(number_of_vehilce):
        qq.append(calculate_vehicle_distances(data).iloc[i, 1])
    infeasible_std_vehicle_distance.append(np.std(qq))


# print('-------------------------------------------------\n')
# print('The average number of nodes in feasible graphs is: ', np.mean(feasible_number_of_nodes))
# print('The average number of edges in feasible graphs is: ', np.mean(feasible_number_of_edges))
# print('-------------------------------------------------\n')
# print('The average number of nodes in infeasible graphs is: ', np.mean(infeasible_number_of_nodes))
# print('The average number of edges in infeasible graphs is: ', np.mean(infeasible_number_of_edges))
# print('-------------------------------------------------\n')


# ## Calculate the average number of 'stop' nodes and 'vehicle' nodes
# print('The average number of stop nodes in feasible graphs is: ', np.mean(feasible_stop_nodes))
# print('The average number of vehicle nodes in feasible graphs is: ', np.mean(feasible_vehicle_nodes))
# print('-------------------------------------------------\n')
# print('The average number of stop nodes in infeasible graphs is: ', np.mean(infeasible_stop_nodes))
# print('The average number of vehicle nodes in infeasible graphs is: ', np.mean(infeasible_vehicle_nodes))
# print('-------------------------------------------------\n')

# ## The calculation of the average number of unloading edges across feasible and infeasible graphs
# print('The average number of unloading edges in feasible graphs is: ', np.mean(feasible_unloading_edges))
# print('The average number of unloading edges in infeasible graphs is: ', np.mean(infeasible_unloading_edges))
# print('-------------------------------------------------\n')


# ## The calculation of the average number of loaded vehicles in each stop and the std
# print('The average number of average loaded vehicles in each stop in feasible graphs is: ', np.mean(average_feasible_stop_load_number)) 
# print('The average standard deviation of the number of loaded vehicles in each stop in feasible graphs is: ', np.mean(standard_deviation_feasible_stop_load_number))
# print('-------------------------------------------------\n')
# print('The average number of average loaded vehicles in each stop in infeasible graphs is: ', np.mean(average_infeasible_stop_load_number))
# print('The average standard deviation of the number of loaded vehicles in each stop in infeasible graphs is: ', np.mean(standard_deviation_infeasible_stop_load_number))
# print('-------------------------------------------------\n')


### -------------------------------------------------



#%%#
### Step-4 ######################################################################
### Visualize the distribution of the number of nodes and edges across the dataset


## Plot the distribution of the number of nodes and edges across feasible and infeasible graphs
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

positions = [1, 1.5, 3, 3.5]

vp = ax.violinplot([feasible_number_of_nodes, infeasible_number_of_nodes, feasible_number_of_edges, infeasible_number_of_edges], showmeans=True, showmedians=True, positions=positions)

colors = ['navy', 'darkorange', 'navy', 'darkorange']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])
    vp['bodies'][i].set_edgecolor('black')
    vp['bodies'][i].set_alpha(0.7)

for partname in ('cbars','cmins','cmaxes'):
    vp[partname].set_edgecolor('black')
    vp[partname].set_linewidth(1)

ax.set_xticks([1.25, 3.25])
ax.set_xticklabels(['Number of Nodes', 'Number of Edges'])
# ax.set_title('Distribution of the Number of Nodes and Edges', fontsize=20, fontweight='bold')
ax.set_ylabel('Count', fontsize=20)

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=12)

# Add the legend
custom_lines = [plt.Line2D([0], [0], color='navy', lw=4),
                plt.Line2D([0], [0], color='darkorange', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Soft Infeasible Solution'], loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()






## Plot the distribution of the number of 'stop' nodes and 'vehicle' nodes across feasible and infeasible graphs
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

positions = [1, 1.5, 3, 3.5]

vp = ax.violinplot([feasible_stop_nodes, infeasible_stop_nodes, feasible_vehicle_nodes, infeasible_vehicle_nodes], showmeans=True, showmedians=True, positions=positions)

colors = ['navy', 'darkorange', 'navy', 'darkorange']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])
    vp['bodies'][i].set_edgecolor('black')
    vp['bodies'][i].set_alpha(0.7)

for partname in ('cbars','cmins','cmaxes'):
    vp[partname].set_edgecolor('black')
    vp[partname].set_linewidth(1)

ax.set_xticks([1.25, 3.25])
ax.set_xticklabels(['Number of stops', 'Number of vehicles'])
# ax.set_title('Distribution of the number of stops and vehicles', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of nodes', fontsize=20)

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=12)

custom_lines = [plt.Line2D([0], [0], color='navy', lw=4),
                plt.Line2D([0], [0], color='darkorange', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Soft Infeasible Solution'], loc='upper right')
plt.show()






## Plot the feasible_avg_vehicle_distance and infeasible_avg_vehicle_distance
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
positions = [1, 1.5, 3, 3.5]

vp = ax.violinplot([feasible_avg_vehicle_distance, infeasible_avg_vehicle_distance, feasible_std_vehicle_distance, infeasible_std_vehicle_distance], showmeans=True, showmedians=True, positions=positions)

colors = ['navy', 'darkorange', 'navy', 'darkorange']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])
    vp['bodies'][i].set_edgecolor('black')
    vp['bodies'][i].set_alpha(0.7)

for partname in ('cbars','cmins','cmaxes'):
    vp[partname].set_edgecolor('black')
    vp[partname].set_linewidth(1)

ax.set_xticks([1.25, 3.25])
ax.set_xticklabels(['Average distance of each vehicle on the deck', 'Std distance of each vehicle on the deck'])
# ax.set_title('Distribution of the average distance and std distance of each vehicle on the deck', fontsize=20, fontweight='bold')
ax.set_ylabel('Distance of Stops', fontsize=20)

ax.tick_params(axis='x', rotation=45, labelsize=18)
ax.tick_params(axis='y', labelsize=12)

custom_lines = [plt.Line2D([0], [0], color='navy', lw=4),
                plt.Line2D([0], [0], color='darkorange', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Soft Infeasible Solution'], loc='upper right')
plt.show()





## Plot the distribution of the number of unloading edges across feasible and infeasible graphs
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
vp = ax.violinplot([feasible_unloading_edges, infeasible_unloading_edges], showmeans=True, showmedians=True)

colors = ['navy', 'darkorange', 'navy', 'darkorange']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])
    vp['bodies'][i].set_edgecolor('black')
    vp['bodies'][i].set_alpha(0.7)

for partname in ('cbars','cmins','cmaxes'):
    vp[partname].set_edgecolor('black')
    vp[partname].set_linewidth(1)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Number of unloading in Feasible', 'Number of unloading in Soft Infeasible'])
# ax.set_title('Distribution of the number of unloading edges', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of edges', fontsize=20)

ax.tick_params(axis='x', rotation=45,labelsize=18)
ax.tick_params(axis='y', labelsize=12)

custom_lines = [plt.Line2D([0], [0], color='navy', lw=4),
                plt.Line2D([0], [0], color='darkorange', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Soft Infeasible Solution'], loc='upper right')
plt.show()



## Plot the distribution of the average number of loaded vehicles in each stop
# plt.style.use('_mpl-gallery')
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# vp = ax.violinplot([average_feasible_stop_load_number, average_infeasible_stop_load_number], showmeans=True, showmedians=True)
# ax.set_xticks([1, 2])
# ax.set_xticklabels(['average_feasible_stop_load_number', 'average_infeasible_stop_load_number'])
# ax.set_title('Distribution of the average number of loaded vehicles in each stop')
# ax.set_ylabel('Number of vehicles')
# ax.tick_params(axis='x', rotation=45)
# plt.show()



### -------------------------------------------------



#%%#
### Step-5 ######################################################################

## Distribution of standard_deviation_feasible_stop_load_number and standard_deviation_infeasible_stop_load_number

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
vp = ax.violinplot([standard_deviation_feasible_stop_load_number, standard_deviation_infeasible_stop_load_number], showmeans=True, showmedians=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(['standard_deviation_feasible_stop_load_number', 'standard_deviation_infeasible_stop_load_number'])
ax.set_title('Distribution of the standard deviation of the number of loaded vehicles in each stop for feasible solution and infeasible solution')
ax.set_ylabel('Number of vehicles')
ax.tick_params(axis='x', rotation=45)
plt.show()


plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
vp = ax.violinplot([average_feasible_stop_load_number, average_infeasible_stop_load_number], showmeans=True, showmedians=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(['average_feasible_stop_load_number', 'average_infeasible_stop_load_number'])
ax.set_title('Distribution of the average number of loaded vehicles in each stop for feasible solution and infeasible solution')
ax.set_ylabel('Number of vehicles')
ax.tick_params(axis='x', rotation=45)
plt.show()



# %%


#%%#
import matplotlib.pyplot as plt
import numpy as np

# Data from the provided table
data = {
    'Basic': {
        'Accuracy': 0.85353769,
        'Precision': 0.87902097,
        'Recall': 0.82039439,
        'F1 Score': 0.84830046,
        'FPR': 0.1133573,
        'FNR': 0.1796066
    },
    'Deck Assign': {
        'Accuracy': 0.84705142,
        'Precision': 0.85399136,
        'Recall': 0.85828393,
        'F1 Score': 0.85603938,
        'FPR': 0.13034936,
        'FNR': 0.14171017
    },
    'Deck Co-use': {
        'Accuracy': 0.81642334,
        'Precision': 0.83232339,
        'Recall': 0.79118745,
        'F1 Score': 0.81159574,
        'FPR': 0.15835528,
        'FNR': 0.20881253
    },
    'Hierarchical': {
        'Accuracy': 0.86221323,
        'Precision': 0.85665691,
        'Recall': 0.87024394,
        'F1 Score': 0.86328133,
        'FPR': 0.14572944,
        'FNR': 0.12975064
    }
}

# Extract the metrics and the graph representations
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'FPR', 'FNR']
graph_representations = list(data.keys())

# Set custom y-axis limits for different metrics
y_limits = {
    'Accuracy': (0.7, 0.9),
    'Precision': (0.7, 0.9),
    'Recall': (0.7, 0.9),
    'F1 Score': (0.7, 0.9),
    'FPR': (0.1, 0.2),
    'FNR': (0.1, 0.3)
}

# Create a bar chart for each metric
fig, axs = plt.subplots(3, 2, figsize=(12, 15))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    values = [data[graph][metric] for graph in graph_representations]
    axs[i].bar(graph_representations, values, color='navy')
    axs[i].set_title(f'cv- {metric} Comparison')
    axs[i].set_ylim(y_limits[metric])
    for j, val in enumerate(values):
        axs[i].text(j, val + 0.002, f'{val:.4f}', ha='center', va='bottom')
    axs[i].set_ylabel(metric)
    axs[i].set_xlabel('Graph Representations')

plt.tight_layout()
plt.show()



#%%#
### ROC-AUC curve and Precision-Recall curve ####################################################


# Define the absolute file paths and labels for each graph representation
file_names = [
    '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed//performance_matrix/roc_pr_data_v2.npz',
    '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/performance_matrix/roc_pr_data_v3_3.npz',
    '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/performance_matrix/roc_pr_data_v3_4.npz',
    '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/performance_matrix/roc_pr_data_v5.npz'
]

labels = ['Basic', 'Deck Assign', 'Deck Co-use', 'Hierarchical']  # Adjust these labels as needed
colors = ['Teal', 'darkorange', 'lightblue', 'pink']  # Define different colors for the lines

# Initialize lists to store data for plotting
roc_curves = []
pr_curves = []
roc_aucs = []
pr_aucs = []

# Load data and compute curves for each file
for file_name, label in zip(file_names, labels):
    data = np.load(file_name, allow_pickle=True)
    true_labels = data['true_labels']
    pred_probs = data['pred_probs']

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    roc_curves.append((fpr, tpr))
    roc_aucs.append(roc_auc)

    # Compute Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc = auc(recall, precision)
    pr_curves.append((precision, recall))
    pr_aucs.append(pr_auc)

    data.close()

# Plot ROC-AUC curves
plt.figure(figsize=(10, 8))
for i, (fpr, tpr) in enumerate(roc_curves):
    plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_aucs[i]:.2f})', color=colors[i], linewidth=2)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
# plt.title('ROC-AUC Curves for Different Graph Representations', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot Precision-Recall curves
plt.figure(figsize=(10, 8))
for i, (precision, recall) in enumerate(pr_curves):
    plt.plot(recall, precision, label=f'{labels[i]} (AUC = {pr_aucs[i]:.2f})', color=colors[i], linewidth=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
# plt.title('Precision-Recall Curves for Different Graph Representations', fontsize=16)
plt.legend(loc='lower left', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()






#%%#
### Individual folds acc, FPR, and Precision #####################################################

import matplotlib.pyplot as plt
import seaborn as sns

colors = ['navy', 'darkorange', 'green', 'red']

###################### Acc for each fold for the graph representations
accuracy_data = {
    'Basic': [0.850974, 0.847889, 0.859649, 0.855408, 0.845961, 0.849431, 0.854444, 0.854058, 0.861963, 0.855601],
    'Deck Assign': [0.854637, 0.852901, 0.842105, 0.853094, 0.858878, 0.855215, 0.776278, 0.848082, 0.869096, 0.860228],
    'Deck Co-use': [0.832466, 0.814536, 0.816079, 0.811837, 0.812801, 0.809523, 0.812199, 0.818392, 0.813572, 0.822826],
    'Hierarchical': [0.866397, 0.861926, 0.863312, 0.855408, 0.864276, 0.862541, 0.867746, 0.858492, 0.860806, 0.861191],
}

# Convert the dictionary to a DataFrame for easy plotting with Seaborn
df = pd.DataFrame(accuracy_data)

# Set the plot style
sns.set(style="whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, palette="Set2", showmeans=False)

# Add data points
sns.stripplot(data=df, palette="Set2", jitter=False, size=8, linewidth=1, edgecolor='gray')

# Set labels and title with increased font sizes
# plt.xlabel('Graph Representation', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.title('Variance of Accuracy across Folds for Different Graph Representations', fontsize=16)

# Adjust y-axis limits
plt.ylim([0.725, 0.915])

# Customize x-tick labels
plt.xticks(ticks=range(len(df.columns)), labels=df.columns, fontsize=14)

# Show the plot
plt.show()







###################### FPR data for each fold for the graph representations
fpr_data = {
    'Basic': [0.112779, 0.099391, 0.104758, 0.124905, 0.097532, 0.099726, 0.127349, 0.102033, 0.119420, 0.145685],
    'Deck Assign': [0.161278, 0.133283, 0.166538, 0.141609, 0.140227, 0.149784, 0.565074, 0.171078, 0.129721, 0.144893],
    'Deck Co-use': [0.141262, 0.143945, 0.164620, 0.168565, 0.171171, 0.151704, 0.173863, 0.154202, 0.161770, 0.152415],
    'Hierarchical': [0.147421, 0.160700, 0.149270, 0.135915, 0.161770, 0.132963, 0.154583, 0.147297, 0.143457, 0.123911],
}

# Convert the dictionary to a DataFrame for easy plotting with Seaborn
df_fpr = pd.DataFrame(fpr_data)

# Set the plot style
sns.set(style="whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_fpr, palette="Set2", showmeans=False)

# Add data points
sns.stripplot(data=df_fpr, palette="Set2", jitter=False, size=8, linewidth=1, edgecolor='gray')

# Set labels and title with increased font sizes
# plt.xlabel('Graph Representation', fontsize=18)
plt.ylabel('FPR', fontsize=18)
plt.title('Variance in FPR Across Folds for Different Graph Representations', fontsize=16)

# Adjust y-axis limits
plt.ylim([0, 0.6])

# Customize x-tick labels
plt.xticks(ticks=range(len(df.columns)), labels=df.columns, fontsize=14)

# Show the plot
plt.show()







###################### Precision data for each fold for the graph representations
precision_data = {
    'Basic': [0.878018, 0.886225, 0.886202, 0.866315, 0.893207, 0.891903, 0.866613, 0.887049, 0.873586, 0.861027],
    'Deck Assign': [0.843247, 0.859888, 0.834981, 0.852976, 0.863254, 0.855199, 0.865648, 0.833831, 0.867549, 0.862819],
    'Deck Co-use': [0.850448, 0.839490, 0.827364, 0.819878, 0.827749, 0.839536, 0.819013, 0.835381, 0.826726, 0.846736],
    'Hierarchical': [0.856132, 0.843064, 0.853207, 0.857880, 0.850145, 0.860977, 0.850740, 0.853097, 0.855162, 0.878068],
}

# Convert the dictionary to a DataFrame for easy plotting with Seaborn
df_precision_data = pd.DataFrame(precision_data)

# Set the plot style
sns.set(style="whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_precision_data, palette="Set2", showmeans=False)

# Add data points
sns.stripplot(data=df_precision_data, palette="Set2", jitter=False, size=8, linewidth=1, edgecolor='gray')

# Set labels and title with increased font sizes
# plt.xlabel('Graph Representation', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Variance in Precision Across Folds for Different Graph Representations', fontsize=16)

# Adjust y-axis limits
plt.ylim([0.775, 0.95])

# Customize x-tick labels
plt.xticks(ticks=range(len(df.columns)), labels=df.columns, fontsize=14)

# Show the plot
plt.show()








######################## F1 Score data for each fold for the graph representations

f1_data = {
    'Basic': [0.845121218,	0.837487127,	0.853815261,	0.850418827,	0.839138313,	0.843455602,	0.851055435,	0.846605876,	0.85799286,	0.858918817],
    'Deck Assign': [0.856708476,	0.84917968,	0.84283247, 0.850294695, 0.860624524, 0.857630332, 0.861094876,	0.850303951, 0.867718683, 0.863951961],
    'Deck Co-use': [0.827681935,	0.804312449,	0.811685748,	0.805500199,	0.812222007,	0.804278922,	0.808402042,	0.812425329,	0.807101536,	0.822346801],
    'Hierarchical': [0.868025138,	0.863619048,	0.864461862,	0.852129338,	0.869387755,	0.863592883,	0.870075758,	0.858683096,	0.860131732,	0.862280031]
}

# Convert the dictionary to a DataFrame for easy plotting with Seaborn
df_f1_data = pd.DataFrame(f1_data)

# Set the plot style
sns.set(style="whitegrid")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_f1_data, palette="Set2", showmeans=False)

# Add data points
sns.stripplot(data=df_f1_data, palette="Set2", jitter=False, size=8, linewidth=1, edgecolor='gray')

# Set labels and title with increased font sizes
# plt.xlabel('Graph Representation', fontsize=18)
plt.ylabel('F1 Score', fontsize=18)
plt.title('Variance in F1 Score Across Folds for Different Graph Representations', fontsize=16)

# Adjust y-axis limits
plt.ylim([0.775, 0.9])

# Customize x-tick labels
plt.xticks(ticks=range(len(df.columns)), labels=df.columns, fontsize=14)

# Show the plot
plt.show()




### ---------------------------------------------------------------------------------------------


#%%#
### Processing time for the graph representations ################################################

## Load the data
df = pd.read_excel('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/processing_time/processing_times_v5_HEAT_att_late.xlsx')


plt.style.use('white')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Increase font size for the boxplot
ax.boxplot([df['v2'], df['v3_3'], df['v3_4'], df['v5']], showmeans=True, patch_artist=True, 
           boxprops=dict(facecolor='lightblue', color='navy'), 
           medianprops=dict(color='navy'), 
           meanprops=dict(marker='o', markerfacecolor='darkorange', markeredgecolor='black'))

# Calculate and annotate means
means = [df['v2'].mean(), df['v3_3'].mean(), df['v3_4'].mean(), df['v5'].mean()]
for i, mean in enumerate(means, start=1):
    ax.text(i + 0.0235, mean + 0.02, f'{mean:.4f}', ha='left', va='bottom', fontsize=13, color='black')  # Increased fontsize

# Set axis labels and title with increased font sizes
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Basic', 'Deck Assign', 'Deck Co-use', 'Hierarchical'], fontsize=18)  # Increased fontsize
# ax.set_title('Distribution of Total Processing Time for Each Graph Representation', fontsize=18, fontweight='bold')  # Increased fontsize
ax.set_ylabel('Processing Time (seconds)', fontsize=18)  # Increased fontsize

# Increase font size for tick labels
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([0, 0.6])

plt.show()




### ---------------------------------------------------------------------------------------------


## Load the data
df = pd.read_excel('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/processing_time/processing_times_v5.xlsx')


plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Increase font size for the boxplot
ax.boxplot([df['v2'], df['v3_3'], df['v3_4'], df['v5']], showmeans=True, patch_artist=True, 
           boxprops=dict(facecolor='lightblue', color='navy'), 
           medianprops=dict(color='navy'), 
           meanprops=dict(marker='o', markerfacecolor='darkorange', markeredgecolor='black'))

# Calculate and annotate means
means = [df['v2'].mean(), df['v3_3'].mean(), df['v3_4'].mean(), df['v5'].mean()]
for i, mean in enumerate(means, start=1):
    ax.text(i + 0.0235, mean + 0.01, f'{mean:.4f}', ha='left', va='bottom', fontsize=13, color='black')  # Increased fontsize

# Set axis labels and title with increased font sizes
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Basic', 'Deck Assign', 'Deck Co-use', 'Hierarchical'], fontsize=18)  # Increased fontsize
# ax.set_title('Distribution of Total Processing Time for Each Graph Representation', fontsize=18, fontweight='bold')  # Increased fontsize
ax.set_ylabel('Processing Time (seconds)', fontsize=18)  # Increased fontsize

# Increase font size for tick labels
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([0, 0.05])

plt.show()





# %%
