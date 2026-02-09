# utils.py - VERSIOE ORIGINALE CORRETTA
from Problem import Problem
import networkx as nx
import numpy as np

def compute_cost(problem: Problem, path):
    alpha = problem.alpha
    beta = problem.beta
    graph = problem.graph
    
    total_cost = 0
    w = 0
    
    for i in range(len(path)-1):
        current_city, gold_to_load = path[i]
        next_city = path[i+1][0]
        
        # Raccogli l'oro PRIMA di viaggiare (come da specifica)
        w += gold_to_load
        
        try:
            sp = nx.shortest_path(graph, current_city, next_city, weight='dist')
            d_ij = nx.path_weight(graph, sp, weight='dist')
        except:
            return float('inf')
        
        total_cost += d_ij + (d_ij * alpha * w) ** beta
        
        if next_city == 0:
            w = 0
    
    return total_cost

def get_baseline_path(problem: Problem):
    total_path = []
    graph = problem.graph
    all_paths = nx.single_source_dijkstra_path(graph, source=0, weight="dist")
    
    for dest, path in all_paths.items():
        if dest == 0: continue
        gold_dest = graph.nodes[dest]['gold']
        total_path.extend([(0, 0), (dest, gold_dest), (0, 0)])
    
    return total_path
