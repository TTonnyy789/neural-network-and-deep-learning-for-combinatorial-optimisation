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


## Convert the json data into a graph, suitable for the graph classification
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
                G.add_edge('stop' + str(index), vehicle, action='unload')

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
                G.add_edge('stop' + str(index), vehicle, action='unload')


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
                            G.add_edge('v' + str(i + 1), node_name, action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge('v' + str(i + 1), node_name, action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge('v' + str(i + 1), node_name, action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge('v' + str(i + 1), node_name, action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge('v' + str(i + 1), node_name, action='applicable')
    

    return G


def json_to_graph_v5(data):
    G = nx.MultiDiGraph()


    ## Create nodes including the stops, vehicles, and decks
    ## Stops
    ## For the feature, add the load and unload vehicle in the feature together, if no load or unload, then only add the stop node with the feature of no load and unload
    ## for example for stop 1, there are v1 load and v2 unload, then combine them together in one feature
    for i in range(len(data['route'])):
        if 'load' in data['route'][i] and 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), process=data['route'][i]['load'] + data['route'][i]['unload'])
        elif 'load' in data['route'][i]:
            G.add_node('stop' + str(i+1), process=data['route'][i]['load'])
        elif 'unload' in data['route'][i]:
            G.add_node('stop' + str(i+1), process=data['route'][i]['unload'])
        else:
            ## TODO: define no loading and unloading any vehicle
            G.add_node('stop' + str(i+1), process=None)

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
                G.add_edge('stop' + str(index), vehicle, action='unload')


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
                            G.add_edge('v' + str(i + 1), node_name, action='applicable')
                        else:
                            if isinstance(access_via[0], list):
                                node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                            G.add_edge('v' + str(i + 1), node_name, action='applicable')
                    elif len(access_via) == 2:
                        if isinstance(access_via[0], list):
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + str('v') + str(i + 1)
                        G.add_edge('v' + str(i + 1), node_name, action='applicable')
                        if isinstance(access_via[1], list):
                            node_name = str(access_via[1][1]) + str(access_via[1][0]) + deck + str('v') + str(i + 1)
                        G.add_edge('v' + str(i + 1), node_name, action='applicable')
                else:
                    node_name = deck + str('v') + str(i + 1)
                    G.add_edge('v' + str(i + 1), node_name, action='applicable')
    

    return G



## Create a adjacency matrix from the graph
def graph_to_adjacency_matrix(G):
    A = nx.adjacency_matrix(G)
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



if __name__ == "__main__":

    ## Create a graph from the json data
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
    # print("json to graph v5 - graph 1")
    # plt.figure(figsize=(10, 10))
    # nx.draw(F_1, with_labels=True, font_weight='bold')
    # plt.show()
    # print("------------------------------------------------- \n")
    # print("json to graph v5 - graph 4")
    # plt.figure(figsize=(10, 10))
    # nx.draw(F_4, with_labels=True, font_weight='bold')
    # plt.show()




### ---------------------------------------------------------------------------



#%%#
