#%%#
### Step-1 ######################################################################
### Read the json file


import json
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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
### Step-1-2 ######################################################################
### Simply sketch a feasible solution for one route for example


## Extract the vehicles and their dimensions
vehicles = data_1['vehicles']
vehicles_df = pd.DataFrame.from_dict(vehicles, orient='index')
vehicles_df.index.name = 'vehicle'
## Add a new column for the vehicle name
vehicles_df['vehicle'] = vehicles_df.index

## Extract the decks and their capacities
decks = data_1['transporter']['decks']
decks_df = pd.DataFrame.from_dict(decks, orient='index')
decks_df.index.name = 'deck'


## Extract the stops information from the entire route
route = data_1['route']
route_df = pd.DataFrame(route)


## If a specific vehicle is loaded in stop i and will be unloaded in stop i+1, then this car must be placed in d3(the last in the list of decks)


def distance_calculate(route_df):
    ## Initialize the list for the loading and unloading point for each vehicle
    load_point = []
    unload_point = []
    ## Iterate every row, for the first = the first stop
    for i in range(len(route_df)):
        ## Checking the loading point for each vehicle
        if isinstance(route_df['load'][i], list):
            for j in vehicles_df['vehicle']:
                if j in route_df['load'][i]:
                    load_point.append(i)

        ## Checking teh unloading point for each vehicle 
        if isinstance(route_df['unload'][i], list):
            for k in vehicles_df['vehicle']:
                if k in route_df['unload'][i]:
                    unload_point.append(i)

    ## Calculate the distance, unload point - load point
    distance = []
    for i in range(len(load_point)):
        distance.append(unload_point[i] - load_point[i])
    
    ## Concat the distance into the stops_df, column name called 'distance'
    vehicles_df['distance'] = distance
    stops_distance_df = vehicles_df.drop('vehicle', axis=1)
    return stops_distance_df

## Calculate the distance between the loading and unloading point for each vehicle
stop_1_distance = distance_calculate(route_df)


def deck_assignment_v1(route_df, stop_distance): ## With the continues situation of the deck
    ## Initialize the truck for recording the overall loading status of the decks in each stop
    truck = [3000, 1000, 3000]
    d1 = []
    d2 = []
    d3 = []
    ## Start from the checking of the stop_distance=1's vehicles
    ## Loop for each stop
    for i in range(len(route_df)):

        if isinstance(route_df['load'][i], list):
            
            ## Loop for each vehicle in the stop i
            for j in route_df['load'][i]:

                ## If the vehicle is in the stop i and the distance between the loading and unloading point is 1, prioritize the vehicle to be loaded in d3, then run the loop for the different distance
                for k in range(1, max(stop_distance['distance'])+1):
                    ## Distance start from 1 to the maximum distance
                    if stop_distance.loc[j, 'distance'] == k:
                        d_vehicle = []
                        ## The following list contains the vehicles that can be loaded in the current stop, k start from 1 and add those vehicles into the list to load in the last deck, d3. And move onto the next distance, for the next distance then load the vehicle in the second last deck, d2, and so on
                        d_vehicle.append(j)

                        ## First for the distance 1, dealing with the dimension of the vehicle and deck capacity
                        for m in sorted(d_vehicle):

                            ## Start from the smallest vehicle, then move to the larger vehicle
                            if truck[len(truck)-1] - stop_distance.loc[m, 'dimension'] >= 0:
                                d3.append(m)
                                truck[len(truck)-1] -= stop_distance.loc[m, 'dimension']

                            ## Till D3 is full, then move to D2
                            elif truck[len(truck)-2] - stop_distance.loc[m, 'dimension'] >= 0:
                                d2.append(m)
                                truck[len(truck)-2] -= stop_distance.loc[m, 'dimension']

                            ## Same as above, till D2 is full, then move to D1
                            else:
                                d1.append(m)
                                truck[len(truck)-3] -= stop_distance.loc[m, 'dimension']

    return d1, d2, d3
                    
d1, d2, d3 = deck_assignment_v1(route_df, stop_1_distance)


## In this updated function, consider the loading and unloading situation of the vehicle in each stop
def deck_assignment_v2(route_df, stop_distance):
    truck = [3000, 1000, 3000] 
    d1 = []
    d2 = []
    d3 = []
    
    ## Initialize three lists to record the deck condition for each stop
    d1_vehicle = []
    d2_vehicle = []
    d3_vehicle = []

    ## Loop for each stop 
    for i in range(len(route_df)):

        ## Ensure the correct loop for the next steps
        if isinstance(route_df['unload'][i], list):
            
            ## Start with the "Unloading" in the first stop
            for a in route_df['unload'][i]:

                if a in d1_vehicle:
                    d1_vehicle.remove(a)
                    truck[len(truck)-3] += stop_distance.loc[a, 'dimension']
                    d1.append(a)

                elif a in d2_vehicle:
                    d2_vehicle.remove(a)
                    truck[len(truck)-2] += stop_distance.loc[a, 'dimension']
                    d2.append(a)

                elif a in d3_vehicle:
                    d3_vehicle.remove(a)
                    truck[len(truck)-1] += stop_distance.loc[a, 'dimension']
                    d3.append(a)

        if isinstance(route_df['load'][i], list):

            ## Sort vehicles by their dimension to maximize space usage
            sorted_vehicles = sorted(route_df['load'][i], key=lambda x: stop_distance.loc[x, 'dimension'])
            
            ## First prioritize vehicles with stop distance of 1
            for vehicle in sorted_vehicles:
                vehicle_distance = stop_distance.loc[vehicle, 'distance']
                vehicle_dimension = stop_distance.loc[vehicle, 'dimension']
                
                if vehicle_distance == 1:
                    if truck[2] >= vehicle_dimension:
                        d3_vehicle.append(vehicle)
                        truck[2] -= vehicle_dimension
                    elif truck[1] >= vehicle_dimension:
                        d2_vehicle.append(vehicle)
                        truck[1] -= vehicle_dimension
                    elif truck[0] >= vehicle_dimension:
                        d1_vehicle.append(vehicle)
                        truck[0] -= vehicle_dimension
            
            ## Then prioritize vehicles with greater distances
            for vehicle in sorted_vehicles:
                vehicle_distance = stop_distance.loc[vehicle, 'distance']
                vehicle_dimension = stop_distance.loc[vehicle, 'dimension']
                
                if vehicle_distance > 1:
                    if truck[2] >= vehicle_dimension:
                        d3_vehicle.append(vehicle)
                        truck[2] -= vehicle_dimension
                    elif truck[1] >= vehicle_dimension:
                        d2_vehicle.append(vehicle)
                        truck[1] -= vehicle_dimension
                    elif truck[0] >= vehicle_dimension:
                        d1_vehicle.append(vehicle)
                        truck[0] -= vehicle_dimension
                    
    return d1, d2, d3


d1, d2, d3 = deck_assignment_v2(route_df, stop_1_distance)             


## Store the deck assignment into a dictionary
deck_assignment = {'d1': d1, 'd2': d2, 'd3': d3}


### ---------------------------------------------------------------------------



#%%#
### Step-1-3 ######################################################################
### Visualize the deck assignment for each stop


## Visualize the deck assignment for first route in each stop using plt package
## For each stop, will have three different bar represent for the deck d1, d2, d3, with the hight of the bar represent the capacity of the deck

## Each deck will have one bar in one stop, overall there will have 3*n(stops) bars in the plot

## Initialize the frame of the plot
plt.figure(figsize=(22, 8))
plt.title('Deck Assignment for the First Route')
colors = plt.cm.viridis(np.linspace(0, 1, len(route_df['load'][0]) +2))


## For the first stop
s1_vehicle = route_df['load'][0]
## Initialize the height for each deck
d1_height = 0
d2_height = 0
d3_height = 0

plt.bar('d1_s1', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d1_s1', 3000, '3000', ha='center', va='bottom')
plt.bar('d2_s1', 1000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d2_s1', 1000, '1000', ha='center', va='bottom')
plt.bar('d3_s1', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d3_s1', 3000, '3000', ha='center', va='bottom')

## For the vehicle in the first stop
for i, vehicle in enumerate(s1_vehicle):
    if vehicle in d1:
        plt.bar('d1_s1', stop_1_distance.loc[vehicle, 'dimension'], bottom=d1_height, color=colors[i], edgecolor='black')

        ## Add the vehicle name on the top of the bar
        plt.text('d1_s1', d1_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')

        d1_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d2:
        plt.bar('d2_s1', stop_1_distance.loc[vehicle, 'dimension'], bottom=d2_height, color=colors[i], edgecolor='black')

        
        plt.text('d2_s1', d2_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d2_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d3:
        plt.bar('d3_s1', stop_1_distance.loc[vehicle, 'dimension'], bottom=d3_height, color=colors[i], edgecolor='black')
        plt.text('d3_s1', d3_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d3_height += stop_1_distance.loc[vehicle, 'dimension']


## For the second stop
s2_vehicle = route_df['load'][1]
d1_height = 0
d2_height = 0
d3_height = 0

plt.bar('d1_s2', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d1_s2', 3000, '3000', ha='center', va='bottom')
plt.bar('d2_s2', 1000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d2_s2', 1000, '1000', ha='center', va='bottom')
plt.bar('d3_s2', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d3_s2', 3000, '3000', ha='center', va='bottom')

## For the vehicle in the second stop
for i, vehicle in enumerate(s2_vehicle):
    if vehicle in d1:
        plt.bar('d1_s2', stop_1_distance.loc[vehicle, 'dimension'], bottom=d1_height, color=colors[i], edgecolor='black')
        plt.text('d1_s2', d1_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d1_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d2:
        plt.bar('d2_s2', stop_1_distance.loc[vehicle, 'dimension'], bottom=d2_height, color=colors[i], edgecolor='black')
        plt.text('d2_s2', d2_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d2_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d3:
        plt.bar('d3_s2', stop_1_distance.loc[vehicle, 'dimension'], bottom=d3_height, color=colors[i], edgecolor='black')
        plt.text('d3_s2', d3_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d3_height += stop_1_distance.loc[vehicle, 'dimension']

## For the third stop
s3_vehicle = route_df['load'][2]
d1_height = 0
d2_height = 0
d3_height = 0

plt.bar('d1_s3', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d1_s3', 3000, '3000', ha='center', va='bottom')
plt.bar('d2_s3', 1000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d2_s3', 1000, '1000', ha='center', va='bottom')
plt.bar('d3_s3', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d3_s3', 3000, '3000', ha='center', va='bottom')

## For the vehicle in the third stop
for i, vehicle in enumerate(s3_vehicle):
    if vehicle in d1:
        plt.bar('d1_s3', stop_1_distance.loc[vehicle, 'dimension'], bottom=d1_height, color=colors[i], edgecolor='black')
        plt.text('d1_s3', d1_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d1_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d2:
        plt.bar('d2_s3', stop_1_distance.loc[vehicle, 'dimension'], bottom=d2_height, color=colors[i], edgecolor='black')
        plt.text('d2_s3', d2_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d2_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d3:
        plt.bar('d3_s3', stop_1_distance.loc[vehicle, 'dimension'], bottom=d3_height, color=colors[i], edgecolor='black')
        plt.text('d3_s3', d3_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d3_height += stop_1_distance.loc[vehicle, 'dimension']

## For the fourth stop
s4_vehicle = route_df['load'][3]
d1_height = 0
d2_height = 0
d3_height = 0

plt.bar('d1_s4', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d1_s4', 3000, '3000', ha='center', va='bottom')
plt.bar('d2_s4', 1000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d2_s4', 1000, '1000', ha='center', va='bottom')
plt.bar('d3_s4', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
plt.text('d3_s4', 3000, '3000', ha='center', va='bottom')

## For the vehicle in the fourth stop
for i, vehicle in enumerate(s4_vehicle):
    if vehicle in d1:
        plt.bar('d1_s4', stop_1_distance.loc[vehicle, 'dimension'], bottom=d1_height, color=colors[i], edgecolor='black')
        plt.text('d1_s4', d1_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d1_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d2:
        plt.bar('d2_s4', stop_1_distance.loc[vehicle, 'dimension'], bottom=d2_height, color=colors[i], edgecolor='black')
        plt.text('d2_s4', d2_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d2_height += stop_1_distance.loc[vehicle, 'dimension']

    elif vehicle in d3:
        plt.bar('d3_s4', stop_1_distance.loc[vehicle, 'dimension'], bottom=d3_height, color=colors[i], edgecolor='black')
        plt.text('d3_s4', d3_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
        d3_height += stop_1_distance.loc[vehicle, 'dimension']

## Merge all stops into one plot
plt.tight_layout()
plt.show()




#%%#
### Step-1-4 ######################################################################
### Visualize the deck assignment for each stop, version 2
## Version 2 for the deck assignment visualization
plt.figure(figsize=(22, 8))
plt.title('Deck Assignment for the First Route')
colors = plt.cm.viridis(np.linspace(0, 1, len(route_df['load'][0]) +2))

for k in range(len(route_df)-1):
    s_vehicle = route_df['load'][k]
    d1_height = 0
    d2_height = 0
    d3_height = 0

    plt.bar(f'd1_s{k+1}', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
    plt.text(f'd1_s{k+1}', 3000, '3000', ha='center', va='bottom')
    plt.bar(f'd2_s{k+1}', 1000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
    plt.text(f'd2_s{k+1}', 1000, '1000', ha='center', va='bottom')
    plt.bar(f'd3_s{k+1}', 3000, color='None', edgecolor='gray', linewidth=2, linestyle='--')
    plt.text(f'd3_s{k+1}', 3000, '3000', ha='center', va='bottom')
    ## Add a vertical line to identify the stop just a bit after the d3 for each stop
    plt.axvline(x=f'Stop{k+1} Deck', color='black', linestyle='-')


    ## Use isinstance() to check if the value is a list
    if isinstance(s_vehicle, list):
        colors = plt.cm.viridis(np.linspace(0, 1, len(route_df['load'][0]) +2))

        for i, vehicle in enumerate(s_vehicle):
            colors = plt.cm.viridis(np.linspace(0, 1, len(route_df['load'][0]) +2))

            if vehicle in d1:
                plt.bar(f'd1_s{k+1}', stop_1_distance.loc[vehicle, 'dimension'], bottom=d1_height, color=colors[i], edgecolor='black')
                plt.text(f'd1_s{k+1}', d1_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
                d1_height += stop_1_distance.loc[vehicle, 'dimension']

            elif vehicle in d2:
                plt.bar(f'd2_s{k+1}', stop_1_distance.loc[vehicle, 'dimension'], bottom=d2_height, color=colors[i], edgecolor='black')
                plt.text(f'd2_s{k+1}', d2_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
                d2_height += stop_1_distance.loc[vehicle, 'dimension']

            elif vehicle in d3:
                plt.bar(f'd3_s{k+1}', stop_1_distance.loc[vehicle, 'dimension'], bottom=d3_height, color=colors[i], edgecolor='black')
                plt.text(f'd3_s{k+1}', d3_height + stop_1_distance.loc[vehicle, 'dimension'], vehicle, ha='center', va='bottom')
                d3_height += stop_1_distance.loc[vehicle, 'dimension']

plt.tight_layout()
plt.show()


### ---------------------------------------------------------------------------



#%%#
### Step-2-1 ######################################################################
### Convert the json data into the required format, dataframe type 1


## First convert the data into a pandas dataframe, which is a tabular data structure and also is suitable for future graph conversion

## For the previous data example: 
## {'route': [{'load': ['v1', 'v2', 'v3']}, {'unload': ['v2', 'v3', 'v1'], 'load': ['v4']}, {'load': ['v5', 'v6']}, {'unload': ['v4', 'v5'], 'load': ['v7']}, {'unload': ['v6', 'v7']}], 'vehicles': {'v1': {'dimension': 2000}, 'v2': {'dimension': 1150}, 'v3': {'dimension': 1200}, 'v4': {'dimension': 950}, 'v5': {'dimension': 850}, 'v6': {'dimension': 1750}, 'v7': {'dimension': 1450}}, 'transporter': {'total_capacity': 7000, 'decks': {'d1': {'capacity': 3000, 'access_via': [['d2', 'd3']]}, 'd2': {'capacity': 1000, 'access_via': [['d3']]}, 'd3': {'capacity': 3000}}}}

# def json_to_dataframe(data):
#     ## Extract the vehicles and their dimensions
#     vehicles = data['vehicles']
#     vehicles_df = pd.DataFrame.from_dict(vehicles, orient='index')
#     vehicles_df.index.name = 'vehicle'
    
#     ## Extract the decks and their capacities
#     decks = data['transporter']['decks']
#     decks_df = pd.DataFrame.from_dict(decks, orient='index')
#     decks_df.index.name = 'deck'
    
#     ## Extract the route information
#     route = data['route']
#     route_df = pd.DataFrame(route)
    
#     ## Combine the dataframes
#     combined_df = pd.concat([vehicles_df, decks_df, route_df], axis=1)
    
#     return combined_df



# df_1 = json_to_dataframe(data_1)
# df_2 = json_to_dataframe(data_2)
# df_3 = json_to_dataframe(data_3)
# df_4 = json_to_dataframe(data_4)


# ## Print the data in the dataframe
# print("-------------------------------------------------")
# print("Data in the dataframe")
# print("-------------------------------------------------")
# print(df_1)
# print("------------------------------------------------- \n")
# print(df_2)
# print("------------------------------------------------- \n")
# print(df_3)
# print("------------------------------------------------- \n")
# print(df_4)


### ---------------------------------------------------------------------------



#%%#
### Step-2-2 ######################################################################
### Convert the json data into the required format, graph for GNN and the dataframe contain the information of the graph


## Converting function version 2
def convert_json_to_graph(json_data):
    ## Extract the information from the json data
    vehicles = json_data['vehicles']
    transporter = json_data['transporter']
    route = json_data['route']
    
    ## Extract vehicle features
    vehicle_features = []
    vehicle_mapping = {}  
    for i, (v_id, v_data) in enumerate(vehicles.items()):
        vehicle_features.append([v_data['dimension'], i])
        vehicle_mapping[v_id] = i
    
    # Extract deck features with placeholder stop index
    deck_features = []
    deck_mapping = {}  # To keep track of deck indices
    for i, (d_id, d_data) in enumerate(transporter['decks'].items()):
        deck_features.append([d_data['capacity'], -1])
        deck_mapping[d_id] = i + len(vehicle_features)  # Deck indices continue after vehicle indices
    
    # Combine features
    features = vehicle_features + deck_features
    x = torch.tensor(features, dtype=torch.float)
    
    # Initialize empty edge index
    edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create the data object
    data = Data(x=x, edge_index=edge_index)
    
    # Define stops
    stops = []
    for stop_index, stop in enumerate(route):
        load_indices = [vehicle_mapping[v_id] for v_id in stop.get('load', [])]
        unload_indices = [vehicle_mapping[v_id] for v_id in stop.get('unload', [])]
        stops.append({"stop_index": stop_index, "load": load_indices, "unload": unload_indices})
    
    return data, stops


## Convert the json data into the graph data
data_1, stops_1 = convert_json_to_graph(data_1)
data_2, stops_2 = convert_json_to_graph(data_2)
data_3, stops_3 = convert_json_to_graph(data_3)
data_4, stops_4 = convert_json_to_graph(data_4)


## Print the data in the graph, using the route 1 for an example including the data and the stops
print("-------------------------------------------------")
print("Data in the graph")
print("-------------------------------------------------")
print(data_1)
print("------------------------------------------------- \n")
print(stops_1)


### ---------------------------------------------------------------------------




#%%#
## Additional Example
import torch
from torch_geometric.data import Data

# Node features: [dimension, deck, stop]
vehicles = torch.tensor([
    [2000, 0, 0],  # V1 at Stop 1
    [1800, 0, 0],  # V2 at Stop 1
    [2200, 1, 1],  # V3 at Stop 2
    [1500, 2, 2],  # V4 at Stop 3
    [2100, 2, 2]   # V5 at Stop 3
], dtype=torch.float)

# Edge index: connections based on loading/unloading sequence
edge_index = torch.tensor([
    [0, 1, 2, 0, 1, 2, 3, 4],
    [1, 2, 3, 2, 3, 4, 0, 1]
], dtype=torch.long)

# Edge attributes: distances or costs between operations (example values)
edge_attr = torch.tensor([
    1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 2.5, 3.0
], dtype=torch.float)

# Create the data object
data = Data(x=vehicles, edge_index=edge_index, edge_attr=edge_attr)

# Define stops with load/unload operations
stops = [
    {'stop_index': 0, 'load': [0, 1], 'unload': []},         # Stop 1: Load V1, V2
    {'stop_index': 1, 'load': [2], 'unload': [0]},           # Stop 2: Load V3, Unload V1
    {'stop_index': 2, 'load': [3, 4], 'unload': [1, 2]},     # Stop 3: Load V4, V5, Unload V2, V3
]

# Vehicle capacity constraint (example value)
vehicle_capacity = 10000

# Example vehicle capacity (in terms of dimensions)
deck_capacities = [6000, 5000, 7000]  # D1, D2, D3

# Visualize the example data
print("Node features (vehicles):")
print(data.x)

print("Edge indices (connections):")
print(data.edge_index)

print("Edge attributes (costs):")
print(data.edge_attr)

print("Stops information:")
print(stops)


#%%#
## Additional Example 2
import torch
from torch_geometric.data import Data

# Node features: vehicles, decks, and stops
node_features = torch.tensor([
    [2000],  # V1
    [1800],  # V2
    [1500],  # V3
    [5000],  # D1
    [4000],  # D2
    [0],     # S1
    [1],     # S2
    [2],     # S3
], dtype=torch.float)

# Edge index: connections based on loading/unloading constraints and actions
edge_index = torch.tensor([
    [0, 1, 1, 2, 5, 5, 6, 6, 7, 7],  # From nodes (V1, V2, V2, V3, S1, S1, S2, S2, S3, S3)
    [3, 3, 4, 4, 0, 1, 2, 0, 1, 2]   # To nodes (D1, D1, D2, D2, V1, V2, V3, V1, V2, V3)
], dtype=torch.long)

# Edge attributes: distances or costs between operations (example values)
edge_attr = torch.tensor([
    1.0, 1.5, 2.0, 1.0, 1.0, 1.5, 2.0, 1.0, 2.5, 3.0
], dtype=torch.float)

# Create the Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Print the data
print("Node features:")
print(data.x)

print("Edge indices:")
print(data.edge_index)

print("Edge attributes:")
print(data.edge_attr)