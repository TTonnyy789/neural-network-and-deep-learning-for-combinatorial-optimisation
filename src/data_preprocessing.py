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
### Step-2 ######################################################################
### Create a graph from the json data


## json data example: 


## Convert the json data into a graph
def json_to_graph(data):
    G = nx.MultiDiGraph()

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1))

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])

    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3
    for deck in data['transporter']['decks']:
        ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
        if 'access_via' not in data['transporter']['decks'][deck]:
            G.add_node(deck, capacity=data['transporter']['decks'][deck]['capacity'])
            continue

        ## Check via option is multiple or not
        access_via = data['transporter']['decks'][deck]['access_via']

        ## One option via entirely
        if len(access_via) == 1:
            if len(access_via[0]) == 1:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][0]) + deck
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            else:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            
        ## Two options via entirely
        elif len(access_via) == 2:
            if isinstance(access_via[0], list):
                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck
            G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            if isinstance(access_via[1], list):
                node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck
            G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
 

    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')


    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                G.add_edge('stop' + str(index), vehicle, action='load')
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge('stop' + str(index), vehicle, action='unload')


    ## TODO: Request the feasible solution of the loading and unloading list for each vehicle, would be like {'v1': [d1], v2: [d2], ...}, for the purpose to connect the edge between the vehicle and the deck, which is the edge between the vehicle and the deck that the vehicle is assigned to

    # for vehicle in data['vehicles']:
    #     for deck in data['transporter']['decks']:
    #         G.add_edge(deck, vehicle, action='assign', cost=1*len(data['transporter']['decks'][deck]['access_via']))

    return G


def json_to_graph_v2(data):
    G = nx.MultiDiGraph()

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1))

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])

    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3
    for deck in data['transporter']['decks']:
        ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
        if 'access_via' not in data['transporter']['decks'][deck]:
            G.add_node(deck, capacity=data['transporter']['decks'][deck]['capacity'])
            continue

        ## Check via option is multiple or not
        access_via = data['transporter']['decks'][deck]['access_via']

        ## One option via entirely
        if len(access_via) == 1:
            if len(access_via[0]) == 1:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][0]) + deck
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            else:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            
        ## Two options via entirely
        elif len(access_via) == 2:
            if isinstance(access_via[0], list):
                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck
            G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            if isinstance(access_via[1], list):
                node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck
            G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
 

    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')


    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                G.add_edge('stop' + str(index), vehicle, action='load')
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge('stop' + str(index), vehicle, action='unload')

    ## Edges between the vehicles and decks that the vehicles are available to access for example, the v1 is over the capacity of d1, so v1 can not access d1, but v1 can access d2 and d3
    ## Basically, the build the link based on the capacity limitation of the vehicle and the deck
    ## And link the vehicle with the node created which is like v1 -> d321
    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck
                            G.add_edge(vehicle, node_name, action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck
                            G.add_edge(vehicle, node_name, action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck
                        G.add_edge(vehicle, node_name, action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck
                        G.add_edge(vehicle, node_name, action='applicable')
                else:
                    G.add_edge(vehicle, deck, action='applicable')
    

    return G


## Co-use the decks, but the decks are after the vehicles
def json_to_graph_v3(data):
    G = nx.MultiDiGraph()

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1))

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])

    ## Deck with the attributes of capacity, and for the deck, use the access_via option to create the node for the pathway of the deck
    ## For example, if there are three decks, then create three nodes for decks
    for deck in data['transporter']['decks']:
        G.add_node(deck, capacity=data['transporter']['decks'][deck]['capacity'])
    
    ## Create edges between the nodes
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')

    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                G.add_edge('stop' + str(index), vehicle, action='load')
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')

    ## Edge between deck and deck, if the deck has access_via option, then create the edge between the deck and the node created for the pathway of the deck
    ## If d1 has access_via option [d2, d3], then create the edge between d1 and d2, and d1 and d3
    ## If d2 has access_via option [d3], then create the edge between d2 and d3
    ## TODO: the automatic checking edge between deck and deck need to be upgraded
    if len(data['transporter']['decks']) == 3:
        G.add_edge('d1', 'd2', action='via')
        G.add_edge('d2', 'd3', action='via')
    elif len(data['transporter']['decks']) == 4:
        G.add_edge('d1', 'd2', action='via')
        G.add_edge('d2', 'd4', action='via')
        G.add_edge('d1', 'd3', action='via')
        G.add_edge('d3', 'd4', action='via')
    
    ## Edge between vehicle and deck, as long as the dimension is less than deck capacity, then the vehicle is applicable to access the deck
    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(vehicle, deck, action='applicable')

    return G


## V3_2 is co-use the deck across all the vehicles, but the decks are in the middle of stop and vehicles
def json_to_graph_v3_2(data):
    G = nx.MultiDiGraph()

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1))

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])

    ## Deck with the attributes of capacity, and for the deck, use the access_via option to create the node for the pathway of the deck
    ## For example, if there are three decks, then create three nodes for decks
    for deck in data['transporter']['decks']:
        G.add_node(deck, capacity=data['transporter']['decks'][deck]['capacity'])
    
    ## Create edges between the nodes
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')

    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')

    ## Edge between deck and deck, if the deck has access_via option, then create the edge between the deck and the node created for the pathway of the deck
    ## If d1 has access_via option [d2, d3], then create the edge between d1 and d2, and d1 and d3
    ## If d2 has access_via option [d3], then create the edge between d2 and d3
    ## TODO: the automatic checking edge between deck and deck need to be upgraded
    if len(data['transporter']['decks']) == 3:
        G.add_edge('d1', 'd2', action='via')
        G.add_edge('d2', 'd3', action='via')
    elif len(data['transporter']['decks']) == 4:
        G.add_edge('d1', 'd2', action='via')
        G.add_edge('d2', 'd4', action='via')
        G.add_edge('d1', 'd3', action='via')
        G.add_edge('d3', 'd4', action='via')
    
    ## Edge between vehicle and deck, as long as the dimension is less than deck capacity, then the vehicle is applicable to access the deck
    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(deck, vehicle, action='applicable')

    ## Edge between stop and deck, if len(data['transporter']['decks']) == 3, then link the stop with d1, d2, d3, if len(data['transporter']['decks']) == 4, then link the stop with d1, d2, d3, d4
    for i in range(1, len(data['route'])+1):
        if len(data['transporter']['decks']) == 3:
            G.add_edge('stop' + str(i), 'd1', action='load via')
            G.add_edge('stop' + str(i), 'd2', action='load via')
            G.add_edge('stop' + str(i), 'd3', action='load via')
        elif len(data['transporter']['decks']) == 4:
            G.add_edge('stop' + str(i), 'd1', action='load via')
            G.add_edge('stop' + str(i), 'd2', action='load via')
            G.add_edge('stop' + str(i), 'd3', action='load via')
            G.add_edge('stop' + str(i), 'd4', action='load via')

    return G


def json_to_graph_v4(data):
    G = nx.MultiDiGraph()


    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    ## For the feature, add the load and unload vehicle in the feature together, if no load or unload, then only add the stop node with the feature of no load and unload

    for i in range(len(data['route'])):
        if 'load' in data['route'][i] and 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), load=data['route'][i]['load'], unload=data['route'][i]['unload'])
        elif 'load' in data['route'][i]:
            G.add_node('stop' + str(i+1), load=data['route'][i]['load'])
        elif 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), unload=data['route'][i]['unload'])
        else:
            G.add_node('stop' + str(i+1), load=None, unload=None)


    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])


    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3

    ## Since in this version, for each vehicle, they will have their own deck option, for run a loop for each vehicle in the route and according to that to create the corresponding deck node
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
            if 'access_via' not in data['transporter']['decks'][deck]:
                node_name = str(deck) + str('v') +str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                continue

            ## Check via option is multiple or not
            access_via = data['transporter']['decks'][deck]['access_via']

            ## One option via entirely
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                else:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                
            ## Two options via entirely
            elif len(access_via) == 2:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                if isinstance(access_via[1], list):
                    node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])


    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')


    ## Edge between stop and deck, focus on the load stop and vehicle, if v1 is going to load at stop 1, then create the edge between stop 1 and all v1 deck nodes
    ## For stop 1, if first check the load vehicle, if v1 is going to load at stop 1, then link d1v1, d2v1, d3v1 with stop 1
    ## Example, if stop 1 is going to load v1 and v2, then like stop 1 wth v1's deck option and v2;s deck option, for the rest of other do not lik them
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                for i in range(len(data['vehicles'])):
                    if vehicle == 'v' + str(i + 1):
                        for deck in data['transporter']['decks']:
                            if 'access_via' in data['transporter']['decks'][deck]:
                                access_via = data['transporter']['decks'][deck]['access_via']
                                if len(access_via) == 1:
                                    if len(access_via[0]) == 1:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                    else:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                elif len(access_via) == 2:
                                    if isinstance(access_via[0], list):
                                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                                    if isinstance(access_via[1], list):
                                        node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                            else:
                                node_name = deck + str('v') + str(i + 1)
                                G.add_edge('stop' + str(index), node_name, action='load via')
                    

    ## Edge between stop and vehicle, focus on the unload stop and vehicle, if v1 is going to unload at stop 1, then create the edge between stop 1 and v1
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')


    ## Edge between vehicle and deck, as long as the dimension does not violate the deck capacity, then the vehicle is applicable to access its own deck across its own options, if v1 can be allocated to d1, and d3, then create the edge between v1 and d1v1, and v1 and d3v1
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
    

    return G


## V5 is the advanced version of v4, connecting the edge between all vehicles' deck options
def json_to_graph_v5(data):
    G = nx.MultiDiGraph()


    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    ## For the feature, add the load and unload vehicle in the feature together, if no load or unload, then only add the stop node with the feature of no load and unload

    for i in range(len(data['route'])):
        if 'load' in data['route'][i] and 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), load=data['route'][i]['load'], unload=data['route'][i]['unload'])
        elif 'load' in data['route'][i]:
            G.add_node('stop' + str(i+1), load=data['route'][i]['load'])
        elif 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), unload=data['route'][i]['unload'])
        else:
            G.add_node('stop' + str(i+1), load=None, unload=None)


    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])


    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3

    ## Since in this version, for each vehicle, they will have their own deck option, for run a loop for each vehicle in the route and according to that to create the corresponding deck node
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
            if 'access_via' not in data['transporter']['decks'][deck]:
                node_name = str(deck) + str('v') +str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                continue

            ## Check via option is multiple or not
            access_via = data['transporter']['decks'][deck]['access_via']

            ## One option via entirely
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                else:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                
            ## Two options via entirely
            elif len(access_via) == 2:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                if isinstance(access_via[1], list):
                    node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])


    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')


    ## Edge between stop and deck, focus on the load stop and vehicle, if v1 is going to load at stop 1, then create the edge between stop 1 and all v1 deck nodes
    ## For stop 1, if first check the load vehicle, if v1 is going to load at stop 1, then link d1v1, d2v1, d3v1 with stop 1
    ## Example, if stop 1 is going to load v1 and v2, then like stop 1 wth v1's deck option and v2;s deck option, for the rest of other do not lik them
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                for i in range(len(data['vehicles'])):
                    if vehicle == 'v' + str(i + 1):
                        for deck in data['transporter']['decks']:
                            if 'access_via' in data['transporter']['decks'][deck]:
                                access_via = data['transporter']['decks'][deck]['access_via']
                                if len(access_via) == 1:
                                    if len(access_via[0]) == 1:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                    else:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                elif len(access_via) == 2:
                                    if isinstance(access_via[0], list):
                                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                                    if isinstance(access_via[1], list):
                                        node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                            else:
                                node_name = deck + str('v') + str(i + 1)
                                G.add_edge('stop' + str(index), node_name, action='load via')
                    

    ## Edge between stop and vehicle, focus on the unload stop and vehicle, if v1 is going to unload at stop 1, then create the edge between stop 1 and v1
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')


    ## Edge between vehicle and deck, as long as the dimension does not violate the deck capacity, then the vehicle is applicable to access its own deck across its own options, if v1 can be allocated to d1, and d3, then create the edge between v1 and d1v1, and v1 and d3v1
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
    
    ## Edge between vehicle's deck, if the vehicle has access to the deck, then create the edge between the vehicle's deck
    ## For example, for each deck in each vehicle, d3v1 -> d32v1 -> d321v1, and run for every vehicle
    for i in range(len(data['vehicles'])):
        if len(data['transporter']['decks']) == 3:
            G.add_edge('d3v' + str(i + 1), 'd3d2v' + str(i + 1), action='via')
            G.add_edge('d3d2v' + str(i + 1), 'd3d2d1v' + str(i + 1), action='via')
        elif len(data['transporter']['decks']) == 4:
            G.add_edge('d4v' + str(i + 1), 'd4d2v' + str(i + 1), action='via')
            G.add_edge('d4d2v' + str(i + 1), 'd4d2d1v' + str(i + 1), action='via')
            G.add_edge('d4v' + str(i + 1), 'd4d3v' + str(i + 1), action='via')
            G.add_edge('d4d3v' + str(i + 1), 'd4d3d1v' + str(i + 1), action='via')

    return G


def json_to_graph_v6(data):
    G = nx.MultiDiGraph()

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    ## For the feature, add the load and unload vehicle in the feature together, if no load or unload, then only add the stop node with the feature of no load and unload

    for i in range(len(data['route'])):
        if 'load' in data['route'][i] and 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), load=data['route'][i]['load'], unload=data['route'][i]['unload'])
        elif 'load' in data['route'][i]:
            G.add_node('stop' + str(i+1), load=data['route'][i]['load'])
        elif 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), unload=data['route'][i]['unload'])
        else:
            G.add_node('stop' + str(i+1), load=None, unload=None)

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, dimension=data['vehicles'][vehicle]['dimension'])

    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## Stop oriented decks, so for stop 1 there will have d3s1, d32s1, and d321s1, for stop 2, there will have d3s2, d32s2, and d321s2
    ## And for 4 deck route will have d4s1, d42s1, d421s1, d43s1, d432s1, d431s1, d41s1, d412s1, d4123s1
    for i in range(len(data['route']) -1 ):
        for deck in data['transporter']['decks']:
            if 'access_via' not in data['transporter']['decks'][deck]:
                node_name = str(deck) + 's' + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                continue

            access_via = data['transporter']['decks'][deck]['access_via']
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][0]) + deck + 's' + str(i + 1)
                    G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                else:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 's' + str(i + 1)
                    G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
            elif len(access_via) == 2:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 's' + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])
                if isinstance(access_via[1], list):
                    node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + 's' + str(i + 1)
                G.add_node(node_name, capacity=data['transporter']['decks'][deck]['capacity'])

    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')

    ## Edge between stop and deck, focus on the load stop and vehicle, if v1 is going to load at stop 1, then create the edge between stop 1 and all v1 deck nodes
    ## Link stop1 with the decks' name have s1, for stop 2, link stop2 with the decks' name have s2
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                for i in range(len(data['vehicles'])):
                    if vehicle == 'v' + str(i + 1):
                        for deck in data['transporter']['decks']:
                            if 'access_via' in data['transporter']['decks'][deck]:
                                access_via = data['transporter']['decks'][deck]['access_via']
                                if len(access_via) == 1:
                                    if len(access_via[0]) == 1:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][0]) + deck + 's' + str(index)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                    else:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 's' + str(index)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                elif len(access_via) == 2:
                                    if isinstance(access_via[0], list):
                                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 's' + str(index)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                                    if isinstance(access_via[1], list):
                                        node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + 's' + str(index)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                            else:
                                node_name = deck + 's' + str(index)
                                G.add_edge('stop' + str(index), node_name, action='load via')

    ## Edge between stop and vehicle, focus on the unload stop and vehicle, if v1 is going to unload at stop 1, then create the edge between stop 1 and v1
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')


    ## Edge between vehicle and deck, as long as the dimension does not violate the deck capacity, then the vehicle is applicable to access its own deck across its own options, if v1 can be allocated to d1, and d3, then create the edge between v1 and d1v1, and v1 and d3v1
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck + 's' + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 's' + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 's' + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + 's' + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                else:
                    node_name = deck + 's' + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), action='applicable')

    ## Edge between vehicle's deck, if the vehicle has access to the deck, then create the edge between the vehicle's deck
    ## For example, for each deck in each stop oriented deck, d3s1 -> d32s1 -> d321s1, and run for every stop
    for i in range(len(data['route'])):
        if len(data['transporter']['decks']) == 3:
            G.add_edge('d3s' + str(i + 1), 'd3d2s' + str(i + 1), action='via')
            G.add_edge('d3d2s' + str(i + 1), 'd3d2d1s' + str(i + 1), action='via')
        elif len(data['transporter']['decks']) == 4:
            G.add_edge('d4s' + str(i + 1), 'd4d2s' + str(i + 1), action='via')
            G.add_edge('d4d2s' + str(i + 1), 'd4d2d1s' + str(i + 1), action='via')
            G.add_edge('d4s' + str(i + 1), 'd4d3s' + str(i + 1), action='via')
            G.add_edge('d4d3s' + str(i + 1), 'd4d3d1s' + str(i + 1), action='via')
    return G


### -----------------------------------------------------


## Feature for stop=1000000000 or 1, torch_geometric only support fixed attribute graph
def json_to_graph_v5_2(data):
    G = nx.MultiDiGraph()


    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    ## For the feature, add the load and unload vehicle in the feature together, if no load or unload, then only add the stop node with the feature of no load and unload

    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=1)


    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])


    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3

    ## Since in this version, for each vehicle, they will have their own deck option, for run a loop for each vehicle in the route and according to that to create the corresponding deck node
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
            if 'access_via' not in data['transporter']['decks'][deck]:
                node_name = str(deck) + str('v') +str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                continue

            ## Check via option is multiple or not
            access_via = data['transporter']['decks'][deck]['access_via']

            ## One option via entirely
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                else:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                
            ## Two options via entirely
            elif len(access_via) == 2:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                if isinstance(access_via[1], list):
                    node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])


    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')


    ## Edge between stop and deck, focus on the load stop and vehicle, if v1 is going to load at stop 1, then create the edge between stop 1 and all v1 deck nodes
    ## For stop 1, if first check the load vehicle, if v1 is going to load at stop 1, then link d1v1, d2v1, d3v1 with stop 1
    ## Example, if stop 1 is going to load v1 and v2, then like stop 1 wth v1's deck option and v2;s deck option, for the rest of other do not lik them
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                for i in range(len(data['vehicles'])):
                    if vehicle == 'v' + str(i + 1):
                        for deck in data['transporter']['decks']:
                            if 'access_via' in data['transporter']['decks'][deck]:
                                access_via = data['transporter']['decks'][deck]['access_via']
                                if len(access_via) == 1:
                                    if len(access_via[0]) == 1:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                    else:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                elif len(access_via) == 2:
                                    if isinstance(access_via[0], list):
                                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                                    if isinstance(access_via[1], list):
                                        node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                            else:
                                node_name = deck + str('v') + str(i + 1)
                                G.add_edge('stop' + str(index), node_name, action='load via')
                    

    ## Edge between stop and vehicle, focus on the unload stop and vehicle, if v1 is going to unload at stop 1, then create the edge between stop 1 and v1
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')


    ## Edge between vehicle and deck, as long as the dimension does not violate the deck capacity, then the vehicle is applicable to access its own deck across its own options, if v1 can be allocated to d1, and d3, then create the edge between v1 and d1v1, and v1 and d3v1
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
    
    ## Edge between vehicle's deck, if the vehicle has access to the deck, then create the edge between the vehicle's deck
    ## For example, for each deck in each vehicle, d3v1 -> d32v1 -> d321v1, and run for every vehicle
    for i in range(len(data['vehicles'])):
        if len(data['transporter']['decks']) == 3:
            G.add_edge('d3v' + str(i + 1), 'd3d2v' + str(i + 1), action='via')
            G.add_edge('d3d2v' + str(i + 1), 'd3d2d1v' + str(i + 1), action='via')
        elif len(data['transporter']['decks']) == 4:
            G.add_edge('d4v' + str(i + 1), 'd4d2v' + str(i + 1), action='via')
            G.add_edge('d4d2v' + str(i + 1), 'd4d2d1v' + str(i + 1), action='via')
            G.add_edge('d4v' + str(i + 1), 'd4d3v' + str(i + 1), action='via')
            G.add_edge('d4d3v' + str(i + 1), 'd4d3d1v' + str(i + 1), action='via')

    return G


def json_to_graph_v4_2(data):
    G = nx.MultiDiGraph()


    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    ## For the feature, add the load and unload vehicle in the feature together, if no load or unload, then only add the stop node with the feature of no load and unload

    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=1)

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])


    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3

    ## Since in this version, for each vehicle, they will have their own deck option, for run a loop for each vehicle in the route and according to that to create the corresponding deck node
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
            if 'access_via' not in data['transporter']['decks'][deck]:
                node_name = str(deck) + str('v') +str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                continue

            ## Check via option is multiple or not
            access_via = data['transporter']['decks'][deck]['access_via']

            ## One option via entirely
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                else:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                
            ## Two options via entirely
            elif len(access_via) == 2:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                if isinstance(access_via[1], list):
                    node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])


    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')


    ## Edge between stop and deck, focus on the load stop and vehicle, if v1 is going to load at stop 1, then create the edge between stop 1 and all v1 deck nodes
    ## For stop 1, if first check the load vehicle, if v1 is going to load at stop 1, then link d1v1, d2v1, d3v1 with stop 1
    ## Example, if stop 1 is going to load v1 and v2, then like stop 1 wth v1's deck option and v2;s deck option, for the rest of other do not lik them
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                for i in range(len(data['vehicles'])):
                    if vehicle == 'v' + str(i + 1):
                        for deck in data['transporter']['decks']:
                            if 'access_via' in data['transporter']['decks'][deck]:
                                access_via = data['transporter']['decks'][deck]['access_via']
                                if len(access_via) == 1:
                                    if len(access_via[0]) == 1:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                    else:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, action='load via')
                                elif len(access_via) == 2:
                                    if isinstance(access_via[0], list):
                                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                                    if isinstance(access_via[1], list):
                                        node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, action='load via')
                            else:
                                node_name = deck + str('v') + str(i + 1)
                                G.add_edge('stop' + str(index), node_name, action='load via')
                    

    ## Edge between stop and vehicle, focus on the unload stop and vehicle, if v1 is going to unload at stop 1, then create the edge between stop 1 and v1
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')


    ## Edge between vehicle and deck, as long as the dimension does not violate the deck capacity, then the vehicle is applicable to access its own deck across its own options, if v1 can be allocated to d1, and d3, then create the edge between v1 and d1v1, and v1 and d3v1
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), action='applicable')
    

    return G


def json_to_graph_v3_3(data):
    G = nx.MultiDiGraph()

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=1)

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    ## Deck with the attributes of capacity, and for the deck, use the access_via option to create the node for the pathway of the deck
    ## For example, if there are three decks, then create three nodes for decks
    for deck in data['transporter']['decks']:
        G.add_node(deck, representation=data['transporter']['decks'][deck]['capacity'])
    

    ## Create edges between the nodes
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), action='next')

    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                G.add_edge('stop' + str(index), vehicle, action='load')
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), action='unload')

    ## Edge between deck and deck, if the deck has access_via option, then create the edge between the deck and the node created for the pathway of the deck
    ## If d1 has access_via option [d2, d3], then create the edge between d1 and d2, and d1 and d3
    ## If d2 has access_via option [d3], then create the edge between d2 and d3
    ## TODO: the automatic checking edge between deck and deck need to be upgraded
    if len(data['transporter']['decks']) == 3:
        G.add_edge('d1', 'd2', action='via')
        G.add_edge('d2', 'd3', action='via')
    elif len(data['transporter']['decks']) == 4:
        G.add_edge('d1', 'd2', action='via')
        G.add_edge('d2', 'd4', action='via')
        G.add_edge('d1', 'd3', action='via')
        G.add_edge('d3', 'd4', action='via')
    
    ## Edge between vehicle and deck, as long as the dimension is less than deck capacity, then the vehicle is applicable to access the deck
    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(vehicle, deck, action='applicable')

    return G



### -----------------------------------------------------


def json_to_graph_v3_weight(data):
    G = nx.MultiDiGraph()

    ## Initialize the remaining capacity at the start to the total capacity of the transporter
    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity

    ## Create a list to store the remaining capacities at each stop
    stops_capacity = []

    # Calculate the remaining capacity at each stop
    for i, route in enumerate(data['route']):
        if 'unload' in route:
            for vehicle in route['unload']:
                remaining_capacity += data['vehicles'][vehicle]['dimension']
        if 'load' in route:
            for vehicle in route['load']:
                remaining_capacity -= data['vehicles'][vehicle]['dimension']
        stops_capacity.append(remaining_capacity)

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=stops_capacity[i])


    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=stops_capacity[i])

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    ## Deck with the attributes of capacity, and for the deck, use the access_via option to create the node for the pathway of the deck
    ## For example, if there are three decks, then create three nodes for decks
    for deck in data['transporter']['decks']:
        G.add_node(deck, representation=data['transporter']['decks'][deck]['capacity'])
    

    ## Create edges between the nodes
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                G.add_edge('stop' + str(index), vehicle, weight=10, action='load')
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), weight=8, action='unload')

    ## Edge between deck and deck, if the deck has access_via option, then create the edge between the deck and the node created for the pathway of the deck
    ## If d1 has access_via option [d2, d3], then create the edge between d1 and d2, and d1 and d3
    ## If d2 has access_via option [d3], then create the edge between d2 and d3
    ## TODO: the automatic checking edge between deck and deck need to be upgraded
    if len(data['transporter']['decks']) == 3:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd3', weight=0, action='via')
    elif len(data['transporter']['decks']) == 4:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd4', weight=0, action='via')
        G.add_edge('d1', 'd3', weight=0, action='via')
        G.add_edge('d3', 'd4', weight=0, action='via')
    
    ## Edge between vehicle and deck, as long as the dimension is less than deck capacity, then the vehicle is applicable to access the deck
    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(vehicle, deck, weight=5, action='applicable')

    return G



## Co-use deck
def json_to_graph_v3_4_weight(data):
    G = nx.MultiDiGraph()

    ## Initialize the remaining capacity at the start to the total capacity of the transporter
    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity

    ## Create a list to store the remaining capacities at each stop
    stops_capacity = []

    # Calculate the remaining capacity at each stop
    for i, route in enumerate(data['route']):
        if 'unload' in route:
            for vehicle in route['unload']:
                remaining_capacity += data['vehicles'][vehicle]['dimension']
        if 'load' in route:
            for vehicle in route['load']:
                remaining_capacity -= data['vehicles'][vehicle]['dimension']
        stops_capacity.append(remaining_capacity)

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=stops_capacity[i])

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    ## Deck with the attributes of capacity, and for the deck, use the access_via option to create the node for the pathway of the deck
    ## For example, if there are three decks, then create three nodes for decks
    for deck in data['transporter']['decks']:
        G.add_node(deck, representation=data['transporter']['decks'][deck]['capacity'])
    
    
    ## Create edges between the nodes
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    ## Edges between vehicles and stops, the first element of data_1['route'] is stop 1, the second element is stop 2, and so on, connect the stops with the vehicles, use two types of edges, load and unload
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), weight=8, action='unload')

    ## Edge between deck and deck, if the deck has access_via option, then create the edge between the deck and the node created for the pathway of the deck
    ## If d1 has access_via option [d2, d3], then create the edge between d1 and d2, and d1 and d3
    ## If d2 has access_via option [d3], then create the edge between d2 and d3
    ## TODO: the automatic checking edge between deck and deck need to be upgraded
    if len(data['transporter']['decks']) == 3:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd3', weight=0, action='via')
    elif len(data['transporter']['decks']) == 4:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd4', weight=0, action='via')
        G.add_edge('d1', 'd3', weight=0, action='via')
        G.add_edge('d3', 'd4', weight=0, action='via')
    
    ## Edge between vehicle and deck, as long as the dimension is less than deck capacity, then the vehicle is applicable to access the deck
    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(deck, vehicle, weight=5, action='applicable')

    ## Edge between stop and deck, if len(data['transporter']['decks']) == 3, then link the stop with d1, d2, d3, if len(data['transporter']['decks']) == 4, then link the stop with d1, d2, d3, d4
    for i in range(1, len(data['route'])+1):
        if len(data['transporter']['decks']) == 3:
            G.add_edge('stop' + str(i), 'd1', weight=10, action='load via')
            G.add_edge('stop' + str(i), 'd2', weight=10, action='load via')
            G.add_edge('stop' + str(i), 'd3', weight=10, action='load via')
        elif len(data['transporter']['decks']) == 4:
            G.add_edge('stop' + str(i), 'd1', weight=10, action='load via')
            G.add_edge('stop' + str(i), 'd2', weight=10, action='load via')
            G.add_edge('stop' + str(i), 'd3', weight=10, action='load via')
            G.add_edge('stop' + str(i), 'd4', weight=10, action='load via')

    return G



## vehicle oriented deck graph with weight
def json_to_graph_v5_weight(data):
    G = nx.MultiDiGraph()


    ## Initialize the remaining capacity at the start to the total capacity of the transporter
    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity

    ## Create a list to store the remaining capacities at each stop
    stops_capacity = []

    # Calculate the remaining capacity at each stop
    for i, route in enumerate(data['route']):
        if 'unload' in route:
            for vehicle in route['unload']:
                remaining_capacity += data['vehicles'][vehicle]['dimension']
        if 'load' in route:
            for vehicle in route['load']:
                remaining_capacity -= data['vehicles'][vehicle]['dimension']
        stops_capacity.append(remaining_capacity)

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=stops_capacity[i])


    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])


    ## Create the node for the deck, and based on the pathway of the deck, which is via.
    ## For example, there are d3 -> d2 -> d1 pathway, then create a single node for this pathway call it d321, and for pathway d3 -> d2, create a single node for d32, and for d3, create a single node d3

    ## Since in this version, for each vehicle, they will have their own deck option, for run a loop for each vehicle in the route and according to that to create the corresponding deck node
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            ## Firstly check whether the deck has access_via option or not, for example for d3 in route 1 is the latest deck, so it does not have access_via option
            if 'access_via' not in data['transporter']['decks'][deck]:
                node_name = str(deck) + str('v') +str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                continue

            ## Check via option is multiple or not
            access_via = data['transporter']['decks'][deck]['access_via']

            ## One option via entirely
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                else:
                    if isinstance(access_via[0], list):
                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                    G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                
            ## Two options via entirely
            elif len(access_via) == 2:
                if isinstance(access_via[0], list):
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])
                if isinstance(access_via[1], list):
                    node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                G.add_node(node_name, representation=data['transporter']['decks'][deck]['capacity'])


    ## Create edges between the nodes
    ## Edges between the stops, stop1 -> stop2 -> stop3 -> ...
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')


    ## Edge between stop and deck, focus on the load stop and vehicle, if v1 is going to load at stop 1, then create the edge between stop 1 and all v1 deck nodes
    ## For stop 1, if first check the load vehicle, if v1 is going to load at stop 1, then link d1v1, d2v1, d3v1 with stop 1
    ## Example, if stop 1 is going to load v1 and v2, then like stop 1 wth v1's deck option and v2;s deck option, for the rest of other do not lik them
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                for i in range(len(data['vehicles'])):
                    if vehicle == 'v' + str(i + 1):
                        for deck in data['transporter']['decks']:
                            if 'access_via' in data['transporter']['decks'][deck]:
                                access_via = data['transporter']['decks'][deck]['access_via']
                                if len(access_via) == 1:
                                    if len(access_via[0]) == 1:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                                    else:
                                        if isinstance(access_via[0], list):
                                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                        G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                                elif len(access_via) == 2:
                                    if isinstance(access_via[0], list):
                                        node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                                    if isinstance(access_via[1], list):
                                        node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                                    G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                            else:
                                node_name = deck + str('v') + str(i + 1)
                                G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                    

    ## Edge between stop and vehicle, focus on the unload stop and vehicle, if v1 is going to unload at stop 1, then create the edge between stop 1 and v1
    for index, route in enumerate(data['route'], start=1):
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), weight=8, action='unload')


    ## Edge between vehicle and deck, as long as the dimension does not violate the deck capacity, then the vehicle is applicable to access its own deck across its own options, if v1 can be allocated to d1, and d3, then create the edge between v1 and d1v1, and v1 and d3v1
    for i in range(len(data['vehicles'])):
        for deck in data['transporter']['decks']:
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                if 'access_via' in data['transporter']['decks'][deck]:
                    access_via = data['transporter']['decks'][deck]['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
    
    ## Edge between vehicle's deck, if the vehicle has access to the deck, then create the edge between the vehicle's deck
    ## For example, for each deck in each vehicle, d3v1 -> d32v1 -> d321v1, and run for every vehicle
    for i in range(len(data['vehicles'])):
        if len(data['transporter']['decks']) == 3:
            G.add_edge('d3v' + str(i + 1), 'd3d2v' + str(i + 1), weight=0, action='via')
            G.add_edge('d3d2v' + str(i + 1), 'd3d2d1v' + str(i + 1),weight=0, action='via')
        elif len(data['transporter']['decks']) == 4:
            G.add_edge('d4v' + str(i + 1), 'd4d2v' + str(i + 1), weight=0, action='via')
            G.add_edge('d4d2v' + str(i + 1), 'd4d2d1v' + str(i + 1), weight=0, action='via')
            G.add_edge('d4v' + str(i + 1), 'd4d3v' + str(i + 1), weight=0, action='via')
            G.add_edge('d4d3v' + str(i + 1), 'd4d3d1v' + str(i + 1), weight=0, action='via')

    return G



## Pure load and unload relationship graph representation
def json_to_graph_v2_weight(data):
    G = nx.MultiDiGraph()

    ## Initialize the remaining capacity at the start to the total capacity of the transporter
    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity

    ## Create a list to store the remaining capacities at each stop
    stops_capacity = []

    # Calculate the remaining capacity at each stop
    for i, route in enumerate(data['route']):
        if 'unload' in route:
            for vehicle in route['unload']:
                remaining_capacity += data['vehicles'][vehicle]['dimension']
        if 'load' in route:
            for vehicle in route['load']:
                remaining_capacity -= data['vehicles'][vehicle]['dimension']
        stops_capacity.append(remaining_capacity)

    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    for i in range(len(data['route'])):
        G.add_node('stop' + str(i+1), representation=stops_capacity[i])

    ## Vehicles with the attributes of dimension
    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])


    ## Create the edge between nodes
    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    
    ## load 
    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                G.add_edge('stop' + str(index), vehicle, weight=10, action='load')
        if 'unload' in route:
            for vehicle in route['unload']:
                G.add_edge(vehicle, 'stop' + str(index), weight=8, action='unload')
    
    return G




### -----------------------------------------------------      




### ------------------------------------------------------------------
## Graph information extraction


## Create a adjacency matrix from the graph
def adjacency_metric_extract(G):
    A = nx.adjacency_matrix(G, weight='weight')
    return A



## Convert the adjacency_matrix into edge index
def edge_index_extract(G):
    ## convert the adjacency matrix into edge index(torch tensor)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    return edge_index


def edge_weight_extract(G):
    a = nx.adjacency_matrix(G, weight='weight')
    coo_matrix = a.tocoo()
    edge_weight = torch.tensor(coo_matrix.data, dtype=torch.float)
    return edge_weight

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


## Calculate the distance for vehicle using the raw json data
def calculate_vehicle_distances(data):
    vehicles = data['vehicles']
    vehicles_df = pd.DataFrame.from_dict(vehicles, orient='index')
    vehicles_df.index.name = 'vehicle'
    vehicles_df['vehicle'] = vehicles_df.index

    decks = data['transporter']['decks']
    decks_df = pd.DataFrame.from_dict(decks, orient='index')
    decks_df.index.name = 'deck'

    route = data['route']
    route_df = pd.DataFrame(route)

    def distance_calculate(route_df, vehicles_df):
        load_point = []
        unload_point = []
        for i in range(len(route_df)):
            if isinstance(route_df['load'][i], list):
                for j in vehicles_df['vehicle']:
                    if j in route_df['load'][i]:
                        load_point.append(i)
            if isinstance(route_df['unload'][i], list):
                for k in vehicles_df['vehicle']:
                    if k in route_df['unload'][i]:
                        unload_point.append(i)

        distance = [unload_point[i] - load_point[i] for i in range(len(load_point))]
        vehicles_df['distance'] = distance
        stops_distance_df = vehicles_df.drop('vehicle', axis=1)
        return stops_distance_df

    stops_distance_df = distance_calculate(route_df, vehicles_df)
    return stops_distance_df


## Feature extraction for the graph to do the stratified sampling for v3_3, v3_4
def extract_graph_features_v2(G):
    features = {}
    features['num_stops'] = sum(1 for n, attr in G.nodes(data=True) if 'stop' in n)
    features['num_vehicles'] = sum(1 for n, attr in G.nodes(data=True) if 'v' in n )
    return features


## For v5
def extract_graph_features_v3(G):
    features = {}
    features['num_stops'] = sum(1 for n, attr in G.nodes(data=True) if 'stop' in n)
    features['num_vehicles'] = sum(1 for n, attr in G.nodes(data=True) if 'v' in n and 'd' not in n)
    return features


## Feature extractor for json file
def extract_json_features(data):
    features = {}
    features['num_stops'] = len(data['route'])
    features['num_vehicles'] = len(data['vehicles'])
    return features


if __name__ == "__main__":

    ## Create a graph from the json data
    G_1 = json_to_graph(data_1)
    GG_1 = json_to_graph_v2(data_1)
    GGG_1 = json_to_graph_v3(data_1)
    GGGG_1 = json_to_graph_v4(data_1)
    F_1 = json_to_graph_v5(data_1)
    G_2 = json_to_graph(data_2)
    G_3 = json_to_graph(data_3)
    G_4 = json_to_graph(data_4)
    GG_4 = json_to_graph_v2(data_4)
    GGG_4 = json_to_graph_v3(data_4)
    GGGG_4 = json_to_graph_v4(data_4)
    F_4 = json_to_graph_v5(data_4)

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
    print("json to graph v1 - graph 1")
    plt.figure(figsize=(10, 10))
    nx.draw(G_1, with_labels=True, font_weight='bold')
    plt.show()
    print("------------------------------------------------- \n")
    print("json to graph v1 - graph 4")
    plt.figure(figsize=(10, 10))
    nx.draw(G_4, with_labels=True, font_weight='bold')
    plt.show()
    print("------------------------------------------------- \n")
    print("json to graph v2 - graph 1")
    plt.figure(figsize=(10, 10))
    nx.draw(GG_1, with_labels=True, font_weight='bold')
    plt.show()
    print("------------------------------------------------- \n")
    print("json to graph v2 - graph 4")
    plt.figure(figsize=(10, 10))
    nx.draw(GG_4, with_labels=True, font_weight='bold')
    plt.show()
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
    print("json to graph v5 - graph 1")
    plt.figure(figsize=(10, 10))
    nx.draw(F_1, with_labels=True, font_weight='bold')
    plt.show()
    print("------------------------------------------------- \n")
    print("json to graph v5 - graph 4")
    plt.figure(figsize=(10, 10))
    nx.draw(F_4, with_labels=True, font_weight='bold')
    plt.show()




### ---------------------------------------------------------------------------



#%%#
