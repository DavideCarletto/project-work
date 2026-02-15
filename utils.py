# utils.py
from Problem import Problem
import networkx as nx
import numpy as np
from typing import List, Tuple

import networkx as nx

def compute_cost(problem, path):
    alpha = problem.alpha
    beta = problem.beta
    graph = problem.graph
    
    total_cost = 0.0
    w = 0.0
    
    if path and path[0][0] != 0:
        path = [(0, 0)] + list(path)
    
    for i in range(len(path) - 1):
        u, gold_to_load = path[i]
        v, _ = path[i+1] 
        
        if u == 0:
            w = 0.0
            
        w += gold_to_load 
        
        d_ij = graph[u][v]['dist']
        
        if w == 0:
             total_cost += d_ij
        else:
             total_cost += d_ij + (d_ij * alpha * w) ** beta
            
    return total_cost

def check_feasibility(
    problem: Problem,
    path: List[Tuple[int, float]],
) -> bool:
    
    graph = problem.graph
    
    if not path:
        return False
    
    # Check edge from base to first city
    if not graph.has_edge(0, path[0][0]):
        print(f"Feasibility failed: no edge between base (0) and first city {path[0][0]}")
        return False
    
    # Check edges between consecutive cities
    for (n1, _), (n2, _) in zip(path, path[1:]):
        if not graph.has_edge(n1, n2):
            print(f"Feasibility failed: no edge between {n1} and {n2}")
            return False
    
    # Check all gold is collected
    gold_at = nx.get_node_attributes(graph, "gold")
    gold_collected = {}
    
    for city, gold in path:
        if gold > 0:
            gold_collected[city] = gold_collected.get(city, 0.0) + gold
    
    for city in graph.nodes():
        if city == 0:
            continue
        expected = gold_at.get(city, 0.0)
        collected = gold_collected.get(city, 0.0)
        if abs(expected - collected) > 1e-4:
            print(f"Feasibility failed: city {city} has {expected:.2f} gold, collected {collected:.2f}")
            return False
    
    return True

def get_baseline_path(problem: Problem):
    total_path = []
    graph = problem.graph
    all_paths = nx.single_source_dijkstra_path(graph, source=0, weight="dist")
    
    sorted_dests = [n for n in graph.nodes if n != 0]
    
    for dest in sorted_dests:
        path_nodes = all_paths[dest] 
        gold_dest = graph.nodes[dest]['gold']
        
        for node in path_nodes[1:-1]:
            total_path.append((node, 0))
            
        total_path.append((dest, gold_dest))
        
        path_return = path_nodes[::-1]
        
        for node in path_return[1:]:
            total_path.append((node, 0))
            
    return total_path
