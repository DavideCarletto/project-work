# Project Work - Computational Intelligence

## Disclaimer

During the development of this project I discussed ideas and possible approaches with the following students:

- Irene Bartolini - s345905
- Michele Carena - s349483
- Alessandro Benvenuti - s343748

However, the code in `s339425.py` was written entirely by me. We shared high-level thoughts about the problem structure and possible strategies, but the actual implementation is my own work.

---

## Solution Strategy

The key observation is that the behavior of the cost function changes drastically depending on the value of `beta`. When `beta` is high (>= 1.5), the penalty for carrying gold grows super-linearly, so it becomes very expensive to carry large loads. When `beta` is low (< 1.5), the penalty is more moderate and it pays off to chain multiple cities together in a single trip instead of going back to the base every time.

Based on this, the solution is split into two strategies.

### Strategy A: High Beta (>= 1.5) - Chunked Star Routes

When `beta >= 1.5`, the cost explodes with the weight carried. The baseline already does star routes (base -> city -> base for each city), but it picks up all the gold in one trip. The idea here is to split each city's gold into `k` smaller chunks and make `k` trips instead of one.

For each city, we search for the optimal number of trips `k` that minimizes the total cost. The cost for `k` trips to a city at distance `d` with total gold `g` is:

```python
chunk = total_gold / k
cost_out = dist_base + (dist_base * alpha * 0)**beta   # going empty
cost_ret = dist_base + (dist_base * alpha * chunk)**beta  # returning loaded
total = k * (cost_out + cost_ret)
```

We test several values of `k` around an initial estimate and pick the one that gives the lowest total cost. To handle large instances (N=1000), we set a dynamic budget for `k`:

```python
ops_budget = 10000
limit_k = max(5, int(ops_budget / num_targets))
```

This ensures we don't waste too much time searching for optimal `k` when there are many cities. Cities are processed in order of distance from the base (closest first), which is a minor optimization.

There is also a special case: when `alpha = 0`, the weight penalty vanishes entirely and the problem reduces to a pure shortest-path problem. In this case, we skip Strategy A and fall through to Strategy B regardless of beta.

### Strategy B: Low Beta (< 1.5) - Clarke-Wright Savings Algorithm

When `beta < 1.5` (or `alpha = 0`), it becomes beneficial to visit multiple cities in a single trip before returning to the base, because the penalty for accumulated weight is moderate. The challenge is deciding which cities to group together.

We use the **Clarke-Wright Savings** heuristic, which is a classic algorithm for Vehicle Routing Problems. The idea is:

1. **Start from the baseline**: assume each city is served independently (star routes).
2. **Compute savings**: for each pair of cities `(u, v)`, calculate how much we save by merging their routes into one (base -> u -> v -> base) instead of two separate trips.
3. **Greedily merge**: sort savings in descending order, and merge routes as long as the merge is valid.

#### Step 1: Baseline Costs

For each city, we compute the baseline star-route cost:

```python
star_costs[n] = cost_out + cost_ret
```

where `cost_out = d + (d * alpha * 0)^beta` (going empty) and `cost_ret = d + (d * alpha * gold)^beta` (returning loaded).

#### Step 2: Savings Computation

For each pair of cities, we compute the saving of merging them. We consider both orderings (u then v, and v then u) because the cost depends on which city we visit first (since we accumulate gold along the way):

```python
# Cost of visiting u then v in one trip
cost_leg_uv = d_uv + (d_uv * alpha * g_u)**beta
cost_merged_uv = (d_0u + base_out_cost) + cost_leg_uv + (d_0v + base_in_term_u)

# Saving compared to two separate star routes
sav_uv = (star_costs[u] + star_costs[v]) - cost_merged_uv
```

For small instances (N < 300), we compute exact shortest-path distances between city pairs using `nx.shortest_path_length`. For larger instances, we fall back on Euclidean distances computed with `scipy.spatial.distance.cdist` to keep runtime manageable.

#### Step 3: Greedy Merging

We sort all positive savings in descending order and greedily merge routes. A merge of `(u, v)` is valid only if:

- `u` is the tail of its current route (no outgoing connection yet)
- `v` is the head of its current route (no incoming connection yet)
- `u` and `v` belong to different routes (to avoid cycles)

```python
if next_node[u] is None and prev_node[v] is None and route_id[u] != route_id[v]:
    next_node[u] = v
    prev_node[v] = u
    # propagate route ID
```

When a merge happens, we update the route identifiers so that all nodes in the absorbed route share the same ID.

#### Step 4: Path Construction

After merging, we reconstruct the final path by identifying route heads (nodes with no predecessor) and following the chain of `next_node` pointers:

```python
start_nodes = [n for n in nodes if prev_node[n] is None]

for start_n in start_nodes:
    chain = []
    curr = start_n
    while curr is not None:
        chain.append(curr)
        curr = next_node[curr]
    # base -> chain -> base
```

Each chain becomes a trip: the thief leaves the base, visits all cities in the chain collecting their gold, and returns to the base before starting the next chain.

### Distance Precomputation

Both strategies rely on knowing the shortest-path distances from the base to all cities. We compute these once at the beginning using Dijkstra's algorithm from NetworkX:

```python
base_lengths = nx.single_source_dijkstra_path_length(graph, 0, weight='dist')
d_0 = np.array([base_lengths[n] for n in nodes])
```

This is much faster than calling `nx.shortest_path_length` individually for each node.

## How to Test

The testing notebook is located at `src/solution.ipynb`. To run it:

1. Make sure to install the required dependencies from `base_requirements.txt`:
   ```
   pip install -r base_requirements.txt
   ```
   You will also need `scipy` and `pandas`:
   ```
   pip install scipy pandas
   ```

2. Open `src/solution.ipynb` in Jupyter or VS Code.

3. The notebook contains the following sections:
   - **Setup and Import**: adds the parent directory to `sys.path` so that `Problem.py`, `utils.py`, and `s339425.py` can be imported.
   - **Verify utility function**: checks that the custom `compute_cost` function in `utils.py` matches the baseline cost computed by `Problem.baseline()`. This is important because we need a reliable cost function to evaluate our solution.
   - **Test solution**: runs the solution on a grid of configurations varying `n_cities`, `density`, `alpha`, and `beta`. For each configuration, it compares the cost of our solution against the baseline and prints whether we win, fail, or tie. Results are saved to `test_results.csv`.

4. Run all cells in order. The output will show a table with the improvement percentage for each configuration.

## Repository Structure

```
Problem.py              # Problem class (provided)
s339425.py              # My solution
utils.py                # Cost computation and baseline path utilities
base_requirements.txt   # Base dependencies
README.md               # This file
src/
    solution.ipynb      # Testing notebook
    test_results.csv    # Output of the test (generated)
```