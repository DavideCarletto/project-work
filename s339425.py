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
    
    # Distances and shortest paths from base
    try:
        base_lengths, base_paths = nx.single_source_dijkstra(graph, 0, weight='dist')
        d_0 = np.array([base_lengths[n] for n in nodes])
    except:
        d_0 = np.array([nx.shortest_path_length(graph, 0, n, weight='dist') for n in nodes])
        base_paths = {n: nx.shortest_path(graph, 0, n, weight='dist') for n in [0] + nodes}

    # Cache for inter-city shortest paths
    _sp_cache = {}

    def get_shortest_path(src, dst):
        """Get shortest path using pre-computed base paths or cached on-demand."""
        if src == 0:
            return base_paths.get(dst, [0, dst])
        if dst == 0:
            return list(reversed(base_paths.get(src, [0, src])))
        key = (src, dst)
        if key not in _sp_cache:
            try:
                _sp_cache[key] = nx.shortest_path(graph, src, dst, weight='dist')
            except:
                _sp_cache[key] = [src, dst]
        return _sp_cache[key]

    def build_leg(src, dst, gold_at_dst):
        """Build path from src to dst using only direct edges.
        Returns [(intermediate1, 0), ..., (dst, gold_at_dst)] (nodes after src)."""
        if src == dst:
            return [(dst, gold_at_dst)] if gold_at_dst > 0 else []
        sp = get_shortest_path(src, dst)
        result = [(node, 0) for node in sp[1:-1]]
        result.append((dst, gold_at_dst))
        return result

    # If alpha = 0, pure TSP.
    force_tsp = (alpha <= 1e-6)

    # First strategy with hight beat (>= 1.5)
    # Like baseline but we take only a portion of the gold instead of taking all of it
    if beta >= 1.5 and not force_tsp:
        final_path = []
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
            
            # Use pre-computed base_paths directly (avoids repeated shortest_path calls)
            path_out = base_paths.get(real_node, [0, real_node])
            path_ret = list(reversed(path_out))
            for _ in range(best_k):
                if remaining <= 1e-6: break
                take = min(portion, remaining)
                # Outbound: base -> city (gold only at destination)
                for node in path_out[1:-1]:
                    final_path.append((node, 0))
                final_path.append((real_node, take))
                # Return: city -> base (no gold pickup)
                for node in path_ret[1:]:
                    final_path.append((node, 0))
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

        # For large N, restrict to K nearest Euclidean neighbors but use REAL graph distances
        if num_targets >= 300:
            K = 150
            # Pre-compute real shortest path lengths from each node (on-demand with budget)
            _dist_cache = {}
            time_budget = 200  # seconds
            for i in range(num_targets):
                if time.time() - start_time > time_budget: break
                u = nodes[i]
                # K nearest neighbors by Euclidean distance
                neighbors_idx = np.argsort(d_mat[i])[:K+1]
                for j in neighbors_idx:
                    if j == i: continue
                    v = nodes[j]
                    key = (u, v) if u < v else (v, u)
                    if key not in _dist_cache:
                        try:
                            _dist_cache[key] = nx.shortest_path_length(graph, u, v, weight='dist')
                        except:
                            _dist_cache[key] = d_mat[i, j]

            for i in range(num_targets):
                u = nodes[i]
                neighbors_idx = np.argsort(d_mat[i])[:K+1]
                for j in neighbors_idx:
                    if j <= i or j == i: continue
                    v = nodes[j]
                    key = (u, v) if u < v else (v, u)
                    d_uv = _dist_cache.get(key, d_mat[i, j])

                    d_0u = d_0[i]
                    d_0v = d_0[j]
                    g_u = golds[i]
                    g_v = golds[j]

                    cost_leg_uv = d_uv + (d_uv * alpha * g_u)**beta
                    cost_leg_vu = d_uv + (d_uv * alpha * g_v)**beta

                    base_out_cost = (d_0u * alpha * 0)**beta
                    base_in_term_u = (d_0v * alpha * (g_u + g_v))**beta
                    base_in_term_v = (d_0u * alpha * (g_v + g_u))**beta

                    cost_merged_uv = (d_0u + base_out_cost) + cost_leg_uv + (d_0v + base_in_term_u)
                    sav_uv = (star_costs[u] + star_costs[v]) - cost_merged_uv

                    cost_merged_vu = (d_0v + base_out_cost) + cost_leg_vu + (d_0u + base_in_term_v)
                    sav_vu = (star_costs[u] + star_costs[v]) - cost_merged_vu

                    if sav_uv > 1e-9:
                        savings.append((sav_uv, u, v, 'uv'))
                    if sav_vu > 1e-9:
                        savings.append((sav_vu, v, u, 'vu'))
        else:
            # N is small: compute all pairs with real shortest paths
            for i in range(num_targets):
                for j in range(i + 1, num_targets):
                    u = nodes[i]
                    v = nodes[j]

                    try:
                        d_uv = nx.shortest_path_length(graph, u, v, weight='dist')
                    except:
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
                    cost_merged_uv = (d_0u + base_out_cost) + \
                                     cost_leg_uv + \
                                     (d_0v + base_in_term_u)
                    
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
        final_path = []
        # Identifying the heads of the merged paths
        start_nodes = [n for n in nodes if prev_node[n] is None]
        
        def chain_cost(chain):
            """Compute actual cost of a chain route: base -> chain -> base with weight accumulation."""
            cost = 0.0
            w = 0.0
            prev_n = 0
            for nd in chain:
                sp = get_shortest_path(prev_n, nd)
                for k in range(len(sp) - 1):
                    d = graph[sp[k]][sp[k+1]]['dist']
                    cost += d if w < 1e-9 else d + (d * alpha * w)**beta
                w += golds[node_to_id[nd]]
                prev_n = nd
            # Return to base
            sp = get_shortest_path(prev_n, 0)
            for k in range(len(sp) - 1):
                d = graph[sp[k]][sp[k+1]]['dist']
                cost += d if w < 1e-9 else d + (d * alpha * w)**beta
            return cost

        def build_star_path(city):
            """Build baseline star route for a single city: base -> city -> base."""
            path = []
            path.extend(build_leg(0, city, golds[node_to_id[city]]))
            path.extend(build_leg(city, 0, 0))
            return path

        # Build paths using direct edges, with local search validation
        for start_n in start_nodes:
            chain = []
            curr = start_n
            while curr is not None:
                chain.append(curr)
                curr = next_node[curr]

            # Local search: compare chain cost vs sum of individual star routes
            merged_cost = chain_cost(chain)
            star_total = sum(star_costs[n] for n in chain)

            if merged_cost <= star_total:
                # Merged route is better, use it
                prev = 0
                for node in chain:
                    final_path.extend(build_leg(prev, node, golds[node_to_id[node]]))
                    prev = node
                final_path.extend(build_leg(prev, 0, 0))
            else:
                # Chain is worse than baseline: try splitting into smaller sub-chains
                best_path = []
                for c in chain:
                    best_path.extend(build_star_path(c))
                best_cost = star_total

                # Try all possible split points to find the best 2-partition
                if len(chain) > 1 and time.time() - start_time < 280:
                    for split in range(1, len(chain)):
                        left = chain[:split]
                        right = chain[split:]
                        split_cost = chain_cost(left) + chain_cost(right)
                        if split_cost < best_cost:
                            best_cost = split_cost
                            best_path = []
                            # Build left sub-chain
                            prev = 0
                            for node in left:
                                best_path.extend(build_leg(prev, node, golds[node_to_id[node]]))
                                prev = node
                            best_path.extend(build_leg(prev, 0, 0))
                            # Build right sub-chain
                            prev = 0
                            for node in right:
                                best_path.extend(build_leg(prev, node, golds[node_to_id[node]]))
                                prev = node
                            best_path.extend(build_leg(prev, 0, 0))

                final_path.extend(best_path)
            
        return final_path