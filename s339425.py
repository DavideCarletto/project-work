# s339425.py - SOLUZIONE COMPLETA 100% POSITIVO
import networkx as nx
import numpy as np
import random
import time
import math

def solution(problem):
    graph = problem.graph
    alpha = problem.alpha
    beta = problem.beta
    start_time = time.time()
    
    nodes = [n for n in graph.nodes if n != 0]
    num_targets = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for i, n in enumerate(nodes)}
    base_idx = num_targets
    
    dist_matrix = np.full((num_targets + 1, num_targets + 1), 1e9, dtype=np.float32)
    g_to_m = node_to_idx.copy()
    g_to_m[0] = base_idx
    
    path_gen = nx.all_pairs_dijkstra_path_length(graph, weight='dist')
    for u, dists in path_gen:
        if u not in g_to_m: continue
        u_idx = g_to_m[u]
        for v, d in dists.items():
            if v in g_to_m:
                v_idx = g_to_m[v]
                dist_matrix[u_idx, v_idx] = d
    
    golds_dict = nx.get_node_attributes(graph, 'gold')
    gold_arr = np.array([golds_dict[idx_to_node[i]] for i in range(num_targets)])
    
    TIME_LIMIT = 25
    
    def calculate_exact_cost(route_idxs):
        cost = 0.0
        w = 0.0
        curr = base_idx
        for idx in route_idxs:
            d = dist_matrix[curr, idx]
            cost += d + (d * alpha * w) ** beta
            w += gold_arr[idx]
            curr = idx
        d_ret = dist_matrix[curr, base_idx]
        cost += d_ret + (d_ret * alpha * w) ** beta
        return cost
    
    # STRATEGIA: sempre restare STRETTAMENTE MIGLIORE della baseline
    baseline_single_costs = np.array([calculate_exact_cost([i]) for i in range(num_targets)])
    baseline_total = baseline_single_costs.sum()
    
    # GREEDY ULTRA CONSERVATIVO: SOLO merge che migliora GARANTITO
    routes = {i: [i] for i in range(num_targets)}
    route_costs = {i: baseline_single_costs[i] for i in range(num_targets)}
    node_to_route = {i: i for i in range(num_targets)}
    
    neighbors = {}
    for i in range(num_targets):
        dists = dist_matrix[i, :num_targets]
        nearest = np.argsort(dists)[1:8]  # TOP 8 vicini
        neighbors[i] = nearest
    
    improved = True
    iterations = 0
    while improved and (time.time() - start_time) < TIME_LIMIT and iterations < 50:
        improved = False
        iterations += 1
        best_saving = 0.00001  # Soglia minima assoluta
        best_merge = None
        
        for rid_i in list(routes):
            if rid_i not in routes: continue
            route_i = routes[rid_i]
            end_node = route_i[-1]
            
            for next_node in neighbors[end_node]:
                rid_j = node_to_route[next_node]
                if rid_i == rid_j or rid_j not in routes: continue
                
                route_j = routes[rid_j]
                if route_j[0] != next_node: continue
                
                # MAX 2 CITTÀ SOLO PER BETA ALTO
                if beta >= 2 and len(route_i) + len(route_j) > 2: continue
                if beta < 2 and len(route_i) + len(route_j) > 3: continue
                
                merged = route_i + route_j
                new_cost = calculate_exact_cost(merged)
                old_cost = route_costs[rid_i] + route_costs[rid_j]
                saving = old_cost - new_cost
                
                # SOLO SE DAVVERO MIGLIORE (soglia 0.001%)
                if saving > 0.00001 * old_cost and saving > best_saving:
                    best_saving = saving
                    best_merge = (rid_i, rid_j, merged, new_cost)
        
        if best_merge:
            rid_i, rid_j, merged, new_cost = best_merge
            routes[rid_i] = merged
            route_costs[rid_i] = new_cost
            for node in routes[rid_j]:
                node_to_route[node] = rid_i
            del routes[rid_j]
            del route_costs[rid_j]
            improved = True
    
    # BUILD PATH
    final_path = [(0, 0)]
    for route in routes.values():
        for idx in route:
            city = idx_to_node[idx]
            final_path.append((city, golds_dict[city]))
        final_path.append((0, 0))
    
    return final_path
