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

feasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/feasible'

infeasible_data_dir = '/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/100K_instances/infeasible'


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
infeasible_graphs_v2 = []
for file in os.listdir(infeasible_data_dir):
    ## Select the file with json format and file name do not contain solution
    if file.endswith('.json'):
        data = read_json_file(os.path.join(infeasible_data_dir, file))
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
selected_infeasible_graphs = np.random.choice(selected_infeasible_graphs, 2610, replace=False)
selected_infeasible_graphs.tolist()

# Convert feasible graphs to NetworkX format
feasible_graphs = []
for i in range(len(feasible_graphs_v2)):
    data = json_to_graph_v5(feasible_graphs_v2[i])
    feasible_graphs.append(data)


infeasible_graphs = []
for i in range(len(selected_infeasible_graphs)):
    data = json_to_graph_v5(selected_infeasible_graphs[i])
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

colors = ['blue', 'red', 'blue', 'red']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])
    
ax.set_xticks([1.4, 3.4])
ax.set_xticklabels(['Number of Nodes', 'Number of Edges'])
ax.set_title('Distribution of the number of nodes and edges')
ax.set_ylabel('Number of nodes')
ax.tick_params(axis='x', rotation=45)

## Add the legend
custom_lines = [plt.Line2D([0], [0], color='blue', lw=4),
                plt.Line2D([0], [0], color='red', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Infeasible Solution'], loc='upper right')
plt.show()




## Plot the distribution of the number of 'stop' nodes and 'vehicle' nodes across feasible and infeasible graphs
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

positions = [1, 1.5, 3, 3.5]

vp = ax.violinplot([feasible_stop_nodes, infeasible_stop_nodes, feasible_vehicle_nodes, infeasible_vehicle_nodes], showmeans=True, showmedians=True, positions=positions)

colors = ['blue', 'red', 'blue', 'red']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])

ax.set_xticks([1.4, 3.4])
ax.set_xticklabels(['Number of stops', 'Number of vehicles'])
ax.set_title('Distribution of the number of stops and vehicles')
ax.set_ylabel('Number of nodes')
ax.tick_params(axis='x', rotation=45)

custom_lines = [plt.Line2D([0], [0], color='blue', lw=4),
                plt.Line2D([0], [0], color='red', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Infeasible Solution'], loc='upper right')
plt.show()




## Plot the feasible_avg_vehicle_distance and infeasible_avg_vehicle_distance
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
positions = [1, 1.5, 3, 3.5]

vp = ax.violinplot([feasible_avg_vehicle_distance, infeasible_avg_vehicle_distance, feasible_std_vehicle_distance, infeasible_std_vehicle_distance], showmeans=True, showmedians=True, positions=positions)

colors = ['blue', 'red', 'blue', 'red']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])

ax.set_xticks([1.5, 3])
ax.set_xticklabels(['Average distance of each vehicle on the deck', 'Std distance of each vehicle on the deck'])
ax.set_title('Distribution of the average distance and std distance of each vehicle on the deck')
ax.set_ylabel('Distance')
ax.tick_params(axis='x', rotation=45)

custom_lines = [plt.Line2D([0], [0], color='blue', lw=4),
                plt.Line2D([0], [0], color='red', lw=4)]
ax.legend(custom_lines, ['Feasible Solution', 'Infeasible Solution'], loc='upper right')
plt.show()







## Plot the distribution of the number of unloading edges across feasible and infeasible graphs
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
vp = ax.violinplot([feasible_unloading_edges, infeasible_unloading_edges], showmeans=True, showmedians=True)

colors = ['blue', 'red', 'blue', 'red']
for i in range(len(vp['bodies'])):
    vp['bodies'][i].set_facecolor(colors[i])

ax.set_xticks([1, 2])
ax.set_xticklabels(['# of feasible_unloading_edges', '# of infeasible_unloading_edges'])
ax.set_title('Distribution of the number of unloading edges')
ax.set_ylabel('Number of edges')
ax.tick_params(axis='x', rotation=45)
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
