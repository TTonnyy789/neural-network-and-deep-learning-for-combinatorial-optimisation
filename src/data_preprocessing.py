"""
Data Preprocessing for Car Transporter Loading Problem (CTLP)

Converts raw JSON problem instances into NetworkX graphs for GNN input.
Four graph representations are supported, each encoding the CTLP structure
at a different level of deck-access detail:

  Basic         (json_to_graph_v2_weight)   — stops and vehicles only
  Deck Assign   (json_to_graph_v3_weight)   — shared deck hierarchy per route
  Deck Co-use   (json_to_graph_v3_4_weight) — stops load directly onto decks
  Hierarchical  (json_to_graph_v5_weight)   — per-vehicle deck access trees

Each graph uses a 'representation' node attribute (remaining capacity / vehicle
dimension / deck capacity) and integer edge weights to encode relationship
strength for downstream GNN layers.
"""

import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_json_file(file_path: str) -> dict:
    """Load a CTLP instance from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph representations
# ---------------------------------------------------------------------------

def json_to_graph_v2_weight(data: dict) -> nx.MultiDiGraph:
    """Basic representation — stops and vehicles only.

    Nodes:
        stop_i  — remaining transporter capacity after stop i
        v_k     — vehicle dimension

    Edges:
        stop_i → stop_{i+1}  (next,        weight=1)
        stop_i → v_k         (load,         weight=10)
        v_k    → stop_i      (unload,       weight=8)
    """
    G = nx.MultiDiGraph()

    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity
    stops_capacity = []
    for route in data['route']:
        for v in route.get('unload', []):
            remaining_capacity += data['vehicles'][v]['dimension']
        for v in route.get('load', []):
            remaining_capacity -= data['vehicles'][v]['dimension']
        stops_capacity.append(remaining_capacity)

    for i in range(len(data['route'])):
        G.add_node('stop' + str(i + 1), representation=stops_capacity[i])

    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    for index, route in enumerate(data['route'], start=1):
        for v in route.get('load', []):
            G.add_edge('stop' + str(index), v, weight=10, action='load')
        for v in route.get('unload', []):
            G.add_edge(v, 'stop' + str(index), weight=8, action='unload')

    return G


def json_to_graph_v3_weight(data: dict) -> nx.MultiDiGraph:
    """Deck Assign representation — shared deck nodes with access hierarchy.

    Nodes:
        stop_i  — remaining transporter capacity after stop i
        v_k     — vehicle dimension
        d_j     — deck capacity (d1/d2/d3 shared across all vehicles)

    Edges:
        stop_i → stop_{i+1}  (next,        weight=1)
        stop_i → v_k         (load,         weight=10)
        v_k    → stop_i      (unload,       weight=8)
        d_j    → d_{j+1}     (via,          weight=0)   — deck access order
        v_k    → d_j         (applicable,   weight=5)   — if dimension fits
    """
    G = nx.MultiDiGraph()

    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity
    stops_capacity = []
    for route in data['route']:
        for v in route.get('unload', []):
            remaining_capacity += data['vehicles'][v]['dimension']
        for v in route.get('load', []):
            remaining_capacity -= data['vehicles'][v]['dimension']
        stops_capacity.append(remaining_capacity)

    for i in range(len(data['route'])):
        G.add_node('stop' + str(i + 1), representation=stops_capacity[i])

    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    for deck in data['transporter']['decks']:
        G.add_node(deck, representation=data['transporter']['decks'][deck]['capacity'])

    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    for index, route in enumerate(data['route'], start=1):
        for v in route.get('load', []):
            G.add_edge('stop' + str(index), v, weight=10, action='load')
        for v in route.get('unload', []):
            G.add_edge(v, 'stop' + str(index), weight=8, action='unload')

    num_decks = len(data['transporter']['decks'])
    if num_decks == 3:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd3', weight=0, action='via')
    elif num_decks == 4:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd4', weight=0, action='via')
        G.add_edge('d1', 'd3', weight=0, action='via')
        G.add_edge('d3', 'd4', weight=0, action='via')

    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(vehicle, deck, weight=5, action='applicable')

    return G


def json_to_graph_v3_4_weight(data: dict) -> nx.MultiDiGraph:
    """Deck Co-use representation — stops load directly onto shared decks.

    Nodes:
        stop_i  — remaining transporter capacity after stop i
        v_k     — vehicle dimension
        d_j     — deck capacity (shared)

    Edges:
        stop_i → stop_{i+1}  (next,        weight=1)
        v_k    → stop_i      (unload,       weight=8)
        d_j    → d_{j+1}     (via,          weight=0)   — deck access order
        d_j    → v_k         (applicable,   weight=5)   — if dimension fits
        stop_i → d_j         (load via,     weight=10)  — loading through deck
    """
    G = nx.MultiDiGraph()

    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity
    stops_capacity = []
    for route in data['route']:
        for v in route.get('unload', []):
            remaining_capacity += data['vehicles'][v]['dimension']
        for v in route.get('load', []):
            remaining_capacity -= data['vehicles'][v]['dimension']
        stops_capacity.append(remaining_capacity)

    for i in range(len(data['route'])):
        G.add_node('stop' + str(i + 1), representation=stops_capacity[i])

    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    for deck in data['transporter']['decks']:
        G.add_node(deck, representation=data['transporter']['decks'][deck]['capacity'])

    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    for index, route in enumerate(data['route'], start=1):
        for v in route.get('unload', []):
            G.add_edge(v, 'stop' + str(index), weight=8, action='unload')

    num_decks = len(data['transporter']['decks'])
    if num_decks == 3:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd3', weight=0, action='via')
    elif num_decks == 4:
        G.add_edge('d1', 'd2', weight=0, action='via')
        G.add_edge('d2', 'd4', weight=0, action='via')
        G.add_edge('d1', 'd3', weight=0, action='via')
        G.add_edge('d3', 'd4', weight=0, action='via')

    for vehicle in data['vehicles']:
        for deck in data['transporter']['decks']:
            if data['vehicles'][vehicle]['dimension'] <= data['transporter']['decks'][deck]['capacity']:
                G.add_edge(deck, vehicle, weight=5, action='applicable')

    for i in range(1, len(data['route']) + 1):
        for deck in data['transporter']['decks']:
            G.add_edge('stop' + str(i), deck, weight=10, action='load via')

    return G


def _add_vehicle_deck_nodes(G: nx.MultiDiGraph, data: dict, remaining_capacity: float,
                             stops_capacity: list) -> None:
    """Helper: add per-vehicle deck nodes to G for json_to_graph_v5_weight."""
    for i in range(len(data['vehicles'])):
        for deck, deck_data in data['transporter']['decks'].items():
            if 'access_via' not in deck_data:
                node_name = deck + 'v' + str(i + 1)
                G.add_node(node_name, representation=deck_data['capacity'])
                continue

            access_via = deck_data['access_via']
            if len(access_via) == 1:
                if len(access_via[0]) == 1:
                    node_name = str(access_via[0][0]) + deck + 'v' + str(i + 1)
                else:
                    node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 'v' + str(i + 1)
                G.add_node(node_name, representation=deck_data['capacity'])
            elif len(access_via) == 2:
                node_name0 = str(access_via[0][1]) + str(access_via[0][0]) + deck + 'v' + str(i + 1)
                node_name1 = str(access_via[1][1]) + str(access_via[1][0]) + deck + 'v' + str(i + 1)
                G.add_node(node_name0, representation=deck_data['capacity'])
                G.add_node(node_name1, representation=deck_data['capacity'])


def json_to_graph_v5_weight(data: dict) -> nx.MultiDiGraph:
    """Hierarchical representation — per-vehicle deck access trees.

    Each vehicle has its own set of deck nodes forming an access hierarchy
    (e.g. d3v1 → d3d2v1 → d3d2d1v1), capturing the physical layering of
    decks that must be traversed to reach the innermost deck.

    Nodes:
        stop_i         — remaining transporter capacity after stop i
        v_k            — vehicle dimension
        d_j v_k        — deck-j node specific to vehicle k (capacity)
        d_j d_m v_k    — composite deck path node (capacity)

    Edges:
        stop_i → stop_{i+1}  (next,        weight=1)
        stop_i → deck_node   (load via,    weight=10)  — per vehicle at load stops
        v_k    → stop_i      (unload,       weight=8)
        deck_node → v_k      (applicable,   weight=5)  — if dimension fits
        deck_node → deck_node (via,         weight=0)  — within-vehicle hierarchy
    """
    G = nx.MultiDiGraph()

    total_capacity = data['transporter']['total_capacity']
    remaining_capacity = total_capacity
    stops_capacity = []
    for route in data['route']:
        for v in route.get('unload', []):
            remaining_capacity += data['vehicles'][v]['dimension']
        for v in route.get('load', []):
            remaining_capacity -= data['vehicles'][v]['dimension']
        stops_capacity.append(remaining_capacity)

    for i in range(len(data['route'])):
        G.add_node('stop' + str(i + 1), representation=stops_capacity[i])

    for vehicle in data['vehicles']:
        G.add_node(vehicle, representation=data['vehicles'][vehicle]['dimension'])

    _add_vehicle_deck_nodes(G, data, remaining_capacity, stops_capacity)

    for i in range(1, len(data['route'])):
        G.add_edge('stop' + str(i), 'stop' + str(i + 1), weight=1, action='next')

    for index, route in enumerate(data['route'], start=1):
        if 'load' in route:
            for vehicle in route['load']:
                v_idx = int(vehicle[1:]) - 1
                for deck, deck_data in data['transporter']['decks'].items():
                    if 'access_via' not in deck_data:
                        node_name = deck + 'v' + str(v_idx + 1)
                        G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                        continue
                    access_via = deck_data['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            node_name = str(access_via[0][0]) + deck + 'v' + str(v_idx + 1)
                        else:
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 'v' + str(v_idx + 1)
                        G.add_edge('stop' + str(index), node_name, weight=10, action='load via')
                    elif len(access_via) == 2:
                        node_name0 = str(access_via[0][1]) + str(access_via[0][0]) + deck + 'v' + str(v_idx + 1)
                        node_name1 = str(access_via[1][1]) + str(access_via[1][0]) + deck + 'v' + str(v_idx + 1)
                        G.add_edge('stop' + str(index), node_name0, weight=10, action='load via')
                        G.add_edge('stop' + str(index), node_name1, weight=10, action='load via')

        for v in route.get('unload', []):
            G.add_edge(v, 'stop' + str(index), weight=8, action='unload')

    num_vehicles = len(data['vehicles'])
    for i in range(num_vehicles):
        for deck, deck_data in data['transporter']['decks'].items():
            if data['vehicles']['v' + str(i + 1)]['dimension'] <= deck_data['capacity']:
                if 'access_via' in deck_data:
                    access_via = deck_data['access_via']
                    if len(access_via) == 1:
                        if len(access_via[0]) == 1:
                            node_name = str(access_via[0][0]) + deck + 'v' + str(i + 1)
                        else:
                            node_name = str(access_via[0][1]) + str(access_via[0][0]) + deck + 'v' + str(i + 1)
                        G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
                    elif len(access_via) == 2:
                        for via in access_via:
                            if len(via) == 1:
                                node_name = str(via[0]) + deck + 'v' + str(i + 1)
                            else:
                                node_name = str(via[1]) + str(via[0]) + deck + 'v' + str(i + 1)
                            G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')
                else:
                    node_name = deck + 'v' + str(i + 1)
                    G.add_edge(node_name, 'v' + str(i + 1), weight=5, action='applicable')

    num_decks = len(data['transporter']['decks'])
    for i in range(num_vehicles):
        if num_decks == 3:
            G.add_edge('d3v' + str(i + 1), 'd3d2v' + str(i + 1), weight=0, action='via')
            G.add_edge('d3d2v' + str(i + 1), 'd3d2d1v' + str(i + 1), weight=0, action='via')
        elif num_decks == 4:
            G.add_edge('d4v' + str(i + 1), 'd4d2v' + str(i + 1), weight=0, action='via')
            G.add_edge('d4d2v' + str(i + 1), 'd4d2d1v' + str(i + 1), weight=0, action='via')
            G.add_edge('d4v' + str(i + 1), 'd4d3v' + str(i + 1), weight=0, action='via')
            G.add_edge('d4d3v' + str(i + 1), 'd4d3d1v' + str(i + 1), weight=0, action='via')

    return G


# ---------------------------------------------------------------------------
# Graph utility functions
# ---------------------------------------------------------------------------

def node_extract(G: nx.MultiDiGraph) -> list:
    """Return all (node_id, attr_dict) tuples."""
    return list(G.nodes(data=True))


def edge_extract(G: nx.MultiDiGraph) -> list:
    """Return all (src, dst, attr_dict) tuples."""
    return list(G.edges(data=True))


def calculate_vehicle_distances(data: dict) -> pd.DataFrame:
    """Compute the number of stops each vehicle spends on the transporter.

    Returns a DataFrame indexed by vehicle ID with columns:
        dimension  — vehicle size
        distance   — stops between load and unload
    """
    vehicles = data['vehicles']
    vehicles_df = pd.DataFrame.from_dict(vehicles, orient='index')
    vehicles_df.index.name = 'vehicle'
    vehicles_df['vehicle'] = vehicles_df.index

    route_df = pd.DataFrame(data['route'])

    load_point, unload_point = [], []
    for i in range(len(route_df)):
        if isinstance(route_df['load'][i], list):
            for j in vehicles_df['vehicle']:
                if j in route_df['load'][i]:
                    load_point.append(i)
        if isinstance(route_df['unload'][i], list):
            for k in vehicles_df['vehicle']:
                if k in route_df['unload'][i]:
                    unload_point.append(i)

    vehicles_df['distance'] = [unload_point[i] - load_point[i] for i in range(len(load_point))]
    return vehicles_df.drop('vehicle', axis=1)


# ---------------------------------------------------------------------------
# Feature extraction (used for balanced dataset sampling)
# ---------------------------------------------------------------------------

def extract_graph_features_v2(G: nx.MultiDiGraph) -> dict:
    """Extract structural features for v2/v3_3/v3_4 graphs (stratified sampling)."""
    return {
        'num_stops': sum(1 for n in G.nodes() if 'stop' in n),
        'num_vehicles': sum(1 for n in G.nodes() if 'v' in n),
    }


def extract_graph_features_v3(G: nx.MultiDiGraph) -> dict:
    """Extract structural features for v5 graphs (vehicle nodes exclude deck nodes)."""
    return {
        'num_stops': sum(1 for n in G.nodes() if 'stop' in n),
        'num_vehicles': sum(1 for n in G.nodes() if 'v' in n and 'd' not in n),
    }


def extract_json_features(data: dict) -> dict:
    """Extract instance-level features directly from JSON (for fast similarity matching)."""
    return {
        'num_stops': len(data['route']),
        'num_vehicles': len(data['vehicles']),
    }


def select_similar_infeasible(feasible_data: list, infeasible_data: list,
                               n_select: int, random_state: int = 42) -> list:
    """Select infeasible instances whose structural features are closest to feasible ones.

    Uses greedy nearest-neighbour matching in standardised feature space to
    produce a balanced, structurally similar dataset.
    """
    feasible_features = pd.DataFrame([extract_json_features(d) for d in feasible_data])
    infeasible_features = pd.DataFrame([extract_json_features(d) for d in infeasible_data])

    scaler = StandardScaler()
    f_scaled = scaler.fit_transform(feasible_features)
    i_scaled = scaler.transform(infeasible_features)

    selected = []
    remaining_scaled = i_scaled.copy()
    remaining_data = list(infeasible_data)

    for feat in f_scaled:
        distances = cdist([feat], remaining_scaled, 'euclidean').flatten()
        idx = int(np.argmin(distances))
        selected.append(remaining_data.pop(idx))
        remaining_scaled = np.delete(remaining_scaled, idx, axis=0)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(selected), size=min(n_select, len(selected)), replace=False)
    return [selected[i] for i in indices]
