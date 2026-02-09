import networkx as nx
import numpy as np
import time
from scipy.spatial.distance import cdist

def solution(problem):
    graph = problem.graph
    alpha = problem.alpha 
    beta = problem.beta
    start_time = time.time()

    nodes = [n for n in graph.nodes if n != 0]
    num_targets = len(nodes)
    
    node_to_id = {n: i for i, n in enumerate(nodes)}
    golds = np.array([graph.nodes[n]['gold'] for n in nodes])
    pos_map = {n: np.array(graph.nodes[n]['pos']) for n in [0] + nodes}
    
    # Distances from bases
    try:
        base_lengths = nx.single_source_dijkstra_path_length(graph, 0, weight='dist')
        d_0 = np.array([base_lengths[n] for n in nodes])
    except:
        d_0 = np.array([nx.shortest_path_length(graph, 0, n, weight='dist') for n in nodes])

    # If alpha = 0, pure TSP.
    force_tsp = (alpha <= 1e-6)

    # First strategy with hight beat (>= 1.5)
    # Like baseline but we take only a portion of the gold instead of taking all of it
    if beta >= 1.5 and not force_tsp:
        final_path = [(0, 0)]
        indices = np.argsort(d_0) 
        
        for i in indices:
            if time.time() - start_time > 285: break
            
            dist_base = d_0[i]
            total_gold = golds[i]
            if total_gold <= 1e-6: continue

            # dynamic value of k (portion of the gold we take) to handle "big N" scenarios (N=1000)
            ops_budget = 10000 
            limit_k = max(5, int(ops_budget / num_targets))
            
            # k is the number of travel we do for each city, the higher the better. However, for big N we have
            # to be careful and not extend too much this value because of computational cost
            start_k = int(np.ceil(total_gold))
            start_k = min(start_k, limit_k)
            start_k = max(1, start_k)

            best_k = start_k
            best_val = float('inf')
            
            # test some values for k
            low_k = max(1, start_k - 5)
            high_k = min(start_k + 5, limit_k + 5)
            
            # simulate the cost for different k 
            for k in range(low_k, high_k + 1):
                chunk = total_gold / k

                cost_out = dist_base + (dist_base * alpha * 0)**beta
                cost_ret = dist_base + (dist_base * alpha * chunk)**beta
                total = k * (cost_out + cost_ret)
                
                if total < best_val:
                    best_val = total
                    best_k = k
            
            real_node = nodes[i]
            portion = total_gold / best_k
            remaining = total_gold
            
            # append to the final path
            for _ in range(best_k):
                if remaining <= 1e-6: break
                take = min(portion, remaining)
                final_path.append((real_node, take))
                final_path.append((0, 0))
                remaining -= take
                
        return final_path

    # Second strategy: low beta (< 1.5) or alpha = 0 -> CLARKE-WRIGHT SAVINGS
    else:
        # 1. Compute baseline costs
        star_costs = {}
        for i, n in enumerate(nodes):
            dist = d_0[i]
            gold = golds[i]
            
            # Going forward: no gold
            cost_out = dist + (dist * alpha * 0)**beta
            # Going back: gold
            cost_ret = dist + (dist * alpha * gold)**beta
            # Final cost considering baseline
            star_costs[n] = cost_out + cost_ret

        # 2. Compute saving to see if it's better to join two cities instead of going back to the base (baseline)
        savings = []
        coords_all = np.array([pos_map[n] for n in nodes])
        # compute euclidean distance for each pair of cities
        d_mat = cdist(coords_all, coords_all, metric='euclidean')
        # if N is small, we can compute the real shortest path on the graph, otherwise we rely in d_mat which we can compute it instantly
        use_real_dist = (num_targets < 300)

        for i in range(num_targets):
            for j in range(i + 1, num_targets):
                u = nodes[i]
                v = nodes[j]
                
                if use_real_dist:
                    try:
                        # if N is small we can directly compute it with nx.shortest_path
                        d_uv = nx.shortest_path_length(graph, u, v, weight='dist')
                    except:
                        # otherwise, we rely on euclidean distance
                        d_uv = d_mat[i, j]
                else:
                    d_uv = d_mat[i, j]

                d_0u = d_0[i]
                d_0v = d_0[j]
                g_u = golds[i]
                g_v = golds[j]
                
                # Compute the cost to go from u to v
                cost_leg_uv = d_uv + (d_uv * alpha * g_u)**beta
                
                # Compute the cost to go from v to u
                cost_leg_vu = d_uv + (d_uv * alpha * g_v)**beta

                # Compute the cost to go base -> u -> base -> v
                base_out_cost = (d_0u * alpha * 0)**beta
                base_in_term_u = (d_0v * alpha * (g_u + g_v))**beta
                base_in_term_v = (d_0u * alpha * (g_v + g_u))**beta
                
                # First option: base -> u -> v -> base
                # cost = (d_0u + out_term) + cost_leg_uv + (d_0v + in_term)
                cost_merged_uv = (d_0u + base_out_cost) + \
                                 cost_leg_uv + \
                                 (d_0v + base_in_term_u)
                
                # Saving if we merge the cities instead of doing like baseline
                sav_uv = (star_costs[u] + star_costs[v]) - cost_merged_uv
                
                # Second option: base -> v -> u -> base
                cost_merged_vu = (d_0v + base_out_cost) + \
                                 cost_leg_vu + \
                                 (d_0u + base_in_term_v)
                           
                sav_vu = (star_costs[u] + star_costs[v]) - cost_merged_vu
                
                if sav_uv > 1e-9:
                    savings.append((sav_uv, u, v, 'uv')) 
                if sav_vu > 1e-9:
                    savings.append((sav_vu, v, u, 'vu'))

        # we save all the savings into a list and order it from the highest to the lowest
        savings.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Merge
        next_node = {n: None for n in nodes}
        prev_node = {n: None for n in nodes}
        route_id = {n: i for i, n in enumerate(nodes)}
        
        for sav, u, v, direction in savings:
            if time.time() - start_time > 285: break

            # Check if a merge is valid based on 3 strict conditions:
            # 'u' is the tail of its route (it has no outgoing connection yet).
            # 'v' is the head of its route (it has no incoming connection yet).
            # They belong to different routes (to prevent creating a closed cycle). 
            if next_node[u] is None and prev_node[v] is None and route_id[u] != route_id[v]:
                next_node[u] = v
                prev_node[v] = u
                # Since we attached v's route to u's route, v's route is now "absorbed". we must update the Route ID for 'v' and all nodes following it to match the ID of 'u'.
                new_id = route_id[u]
                curr = v
                # Propagate the new ID down the chain until the end of the route
                while curr is not None:
                    route_id[curr] = new_id
                    curr = next_node[curr]
                    
        # 4. Build the final path based on the merging
        final_path = [(0, 0)]
        # Identifying the heads of the merged paths
        start_nodes = [n for n in nodes if prev_node[n] is None]
        
        # Going thgrough all the different paths we created and translate in the correct format
        for start_n in start_nodes:
            chain = []
            curr = start_n
            while curr is not None:
                chain.append(curr)
                curr = next_node[curr]
            
            # Base -> Chain -> Base
            for node in chain:
                final_path.append((node, golds[node_to_id[node]]))
            final_path.append((0, 0))

        return final_path