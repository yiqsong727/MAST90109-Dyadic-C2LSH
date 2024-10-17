import numpy as np
from scipy.integrate import quad  
from scipy.spatial.distance import euclidean
import heapq
import time
from numba import jit
from pyinstrument import Profiler
import math
import itertools
from copy import deepcopy

class Node:
    def __init__(self, count, index, ni, layer, config):# config use T_INIT
        self.count = count  # Collision count
        self.index = index  # Node index
        self.ni = ni  # Number of points in this node
        self.layer = layer  # Layer of the node
        self.t = config.T_INIT  # Current t parameter, default is 0
        self.left = None  # Left child node
        self.right = None  # Right child node
        self.parent = None  # Parent node


class DyadicTree:
    def __init__(self, n, indices, config):
        self.n = n
        self.tree = []  # Store nodes of different layers
        self.indices = indices  # Collision point positions
        self.config = config 
        self.build_tree()

    def build_tree(self):
        # Step 1: Initialize the bottom layer nodes (Layer 0)
        #layer = [Node(1 if i in self.indices else 0, i,1, layer index)for i in range(self.n)]
        layer_index=0 
        layer = []
        for i in range(self.n):
            layer.append( Node (0, i, 1, layer_index, self.config))
        for i in range(len(self.indices)):
            n= self.indices[i]
            if n< self.n:
                layer[self.indices[i]].count = 1
        self.tree.append(layer)

        # Step 2: Build dyadic structure layer by layer upwards
        current_layer_size = self.n
        while len(layer) > 1:
            new_layer = []
            layer_index += 1  # Move up layer by layer, increment layer index
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else None  # Consider unpaired cases

                # Create parent node
                parent_index = i // 2
                parent_count = left.count + (right.count if right else 0)
                parent_ni = left.ni + (right.ni if right else 0)
                parent = Node(parent_count, parent_index, parent_ni, layer_index , self.config)

                # Set parent node reference for child nodes
                left.parent = parent
                if right:
                    right.parent = parent

                # Set child nodes for parent node
                parent.left = left
                parent.right = right

                new_layer.append(parent)

            # Add new layer
            self.tree.append(new_layer)
            layer = new_layer
            current_layer_size //= 2

        # After completion, self.tree stores nodes of all layers, with the top layer being the root node

    def print_tree(self):
        
        for i, layer in enumerate(self.tree):
            print(f"Layer {i}: ", end="")
            for node in layer:
                print(f"[count: {node.count}, ni: {node.ni}, index: {node.index}, layer: {node.layer}]", end=", ")
            print()  # New line

    def find_node(self, layer, index):
        
        if layer < len(self.tree):
            for node in self.tree[layer]:
                if node.index == index:
                    return node
        return None


def merge_dyadic_trees(dyadic_trees, p2, m, config, use_heuristic1=True, use_heuristic2=True, use_heuristic3=True ):
    merge_count = 0 
    candidate_set = []
    max_layer = len(dyadic_trees[0].tree) - 1
    heap = []
    max_candidates = int(config.BETA_FP * config.N + config.NEAREST_K )   # Ensure  integer
    merge_threshold = 2*config.N
    
    # Start from the root node
    total_count = sum(tree.tree[max_layer][0].count for tree in dyadic_trees)
    total_ni = dyadic_trees[0].tree[max_layer][0].ni
    t = config.T_INIT
    
    # Check root node with selected heuristic(s)
    if (use_heuristic1 and heuristic_1(total_count, total_ni, p2, m, t, config.ALPHA, config.RATE_BAD_PTS)) or \
       (use_heuristic2 and heuristic_2(total_count, total_ni, p2, m, 0, t, config.ALPHA)) or\
        use_heuristic3:
        heapq.heappush(heap, (max_layer, -total_ni,  0, total_count, t))
        merge_count += 1

    
    while heap:
        current_layer, _,  current_index, current_count, current_t = heapq.heappop(heap)
        merge_count += 1

        if merge_count > merge_threshold and len(candidate_set) < 1:
            #print(f"Early termination: merge_count={merge_count}, candidates={len(candidate_set)}")
            return candidate_set
        
        if current_layer == 0:
            
            
            if current_count > config.ALPHA * m:
                candidate_set.append((current_index, current_count))
                #print(f"Total merge operations triggered: {merge_count}")
                if len(candidate_set) >= max_candidates:
                    #print(f"Total candidate this round: {len(candidate_set)}")
                    return candidate_set
            continue
        
        left_child_total_count = 0
        for tree in dyadic_trees:
            current_layer_nodes = tree.tree[current_layer - 1]
            if current_layer_nodes and 2 * current_index < len(current_layer_nodes):
                left_child_total_count += current_layer_nodes[2 * current_index].count

        right_child_total_count = 0
        for tree in dyadic_trees:
            current_layer_nodes = tree.tree[current_layer - 1]
            if current_layer_nodes and 2 * current_index + 1 < len(current_layer_nodes):
                right_child_total_count += current_layer_nodes[2 * current_index + 1].count
        

        left_child_ni = 0
        first_tree_current_layer = dyadic_trees[0].tree[current_layer - 1]
        if first_tree_current_layer and 2 * current_index < len(first_tree_current_layer):
            left_child_ni = first_tree_current_layer[2 * current_index].ni

        right_child_ni = 0
        first_tree_current_layer = dyadic_trees[0].tree[current_layer - 1]
        if first_tree_current_layer and 2 * current_index + 1 < len(first_tree_current_layer):
            right_child_ni = first_tree_current_layer[2 * current_index + 1].ni
        


        left_satisfies1 = use_heuristic1 and heuristic_1(left_child_total_count, left_child_ni, p2, m, current_t, config.ALPHA, config.RATE_BAD_PTS)
        left_satisfies2 = use_heuristic2 and heuristic_2(left_child_total_count, left_child_ni, p2, m, current_layer - 1, current_t, config.ALPHA)
        left_satisfies3 = use_heuristic3 and heuristic_3(left_child_total_count, current_count, m, current_layer - 1, config )
        right_satisfies1 = use_heuristic1 and heuristic_1(right_child_total_count, right_child_ni, p2, m, current_t, config.ALPHA, config.RATE_BAD_PTS) if right_child_ni > 0 else False
        right_satisfies2 = use_heuristic2 and heuristic_2(right_child_total_count, right_child_ni, p2, m, current_layer - 1, current_t, config.ALPHA) if right_child_ni > 0 else False
        right_satisfies3 = use_heuristic3 and heuristic_3(right_child_total_count, current_count, m, current_layer - 1, config) if right_child_ni > 0 else False
        
        left_satisfies = left_satisfies1 or left_satisfies2 or left_satisfies3
        right_satisfies = right_satisfies1 or right_satisfies2 or right_satisfies3
        
        if left_satisfies and right_satisfies:
            new_t = current_t + 1
        elif left_satisfies or right_satisfies:
            new_t = current_t
        else:
            new_t = current_t - 1
            if new_t <= 0:
                continue  # Abandon this node
        
        if left_satisfies:
            heapq.heappush(heap, (current_layer - 1, -left_child_ni,  2 * current_index, left_child_total_count, new_t))
            
        if right_satisfies:
            heapq.heappush(heap, (current_layer - 1 , -right_child_ni, 2 * current_index + 1, right_child_total_count, new_t))
            
    #print(f"Total merge operations triggered: {merge_count}")
    #print(f"Total candidate this round: {len(candidate_set)}")
    return candidate_set

@jit(nopython=True)
def heuristic_1(node_count, ni, p2, m, t, alpha , rate_bad_pts ):
    #can adjust t
    l = alpha * m
    bad_sum = calculate_bad_sum(p2, alpha , m, rate_bad_pts, ni, t)
    
    numerator = node_count - bad_sum
    denominator = ni - (rate_bad_pts * ni)
    
    # If the denominator is 0, return False directly to avoid division by 0 error
    if abs(denominator) < 1e-10:
        return False
    
    heuristic_value = numerator / denominator
    
    return heuristic_value > l

@jit(nopython=True)
def heuristic_2(node_count, ni, p2, m, i, t, alpha):

    St = calculate_St(p2, alpha, m, ni, t, i)*1.2
    return node_count > St

@jit(nopython=True)
def _heuristic_3_core(node_count, parent_count, m, layer, alpha, g_bias_est, g_std_est, gamma, beta, k):
    if layer == 0:
        return node_count > alpha * m, g_bias_est, g_std_est

    if parent_count <= 1e-6:  # small threshold
        return False, g_bias_est, g_std_est
    
    delta = (node_count - parent_count / 2) / (parent_count / 2) + 10e-6
    
    # update global estimate
    g_bias_est = gamma * delta + (1 - gamma) * g_bias_est
    g_std_est = beta * abs(delta - g_bias_est) + (1 - beta) * g_std_est

    epsilon = k * g_std_est
    Z = (delta - g_bias_est) / g_std_est

    return abs(Z) > epsilon, g_bias_est, g_std_est

def heuristic_3(node_count, parent_count, m, layer, config):
    result, config.G_BIAS_EST, config.G_STD_EST = _heuristic_3_core(
        node_count, parent_count, m, layer, 
        config.ALPHA, config.G_BIAS_EST, config.G_STD_EST, 
        config.GAMMA, config.BETA, config.K
    )
    return result
#--------------------------------------------------------------------------------------------------------------
#------------------------------------------------Calculation---------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
def calculate_m(p1, p2, config):
    
    term1 = (1 / (2 * (p1 - config.ALPHA)**2)) * np.log(1 / config.SIGMA)
    term2 = (1 / (2 * (config.ALPHA - p2)**2)) * np.log(2 / config.BETA_FP)
    return int(np.ceil(max(term1, term2)))

def f2(x):
   
    return (2 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

def p_s(s, w):
    
    result, _ = quad(lambda t: (1 / s) * f2(t / s) * (1 - t / w), 0, w)
    return result

#Heuristic 1
@jit(nopython=True)
def calculate_collision_probability(p2, alpha, m, i):
    
    lower_bound = alpha * m / (2 ** (i + 1))
    upper_bound = alpha * m / (2 ** i)
    
    # Ensure the values are non-negative to avoid issues with sqrt and exponential functions
    lower_diff = max(0, p2 - lower_bound)
    upper_diff = max(0, upper_bound - (1 - p2))
    
    # Calculate the probability lower bound using the given formula
    prob = 1 - np.exp(-2 * m * (lower_diff ** 2)) - np.exp(-2 * m * (upper_diff ** 2))
    
    # Ensure the result is within [0, 1]
    return max(0, min(prob, 1))

#Heuristic 1
@jit(nopython=True)
def calculate_bad_sum(p2, alpha, m, rate_bad_pts, ni, t):
    
    Y = rate_bad_pts * ni  # Estimated number of bad points in the current dyadic node
    sum_value = 0  # Initial value of the sum

    for j in range(1, t + 1):
        # Calculate P_j
        Pj = calculate_collision_probability(p2, alpha, m, j)
        
        # Calculate and add to the total sum
        sum_value += Pj * (alpha * m) / (2 ** (j + 1)) * Y

    return sum_value

#Heuristic 2
@jit(nopython=True)
def calculate_St(p2, alpha, m, ni, t, i):    
    St = 0
    for j in range(1, t + 1):
        Pj = calculate_collision_probability(p2, alpha, m, j)
        St += (ni / (2**i)) * Pj * (alpha * m) / (2**j)
    return St

#--------------------------------------------------------------------------------------------------------------
#------------------------------------------------Implementation------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def generate_bucket_indices(data, q, m, R, config):
    #the input R is the dynamic R
    n, d = data.shape
    W = config.W
    hash_functions = []
    hash_tables = []
    
    for _ in range(m):
        random_vector = np.random.randn(d)
        random_offset = np.random.uniform(0, W)
        hash_func = lambda x, rv=random_vector, ro=random_offset: int(np.floor((np.dot(rv, x) + ro) / W))
        hash_functions.append(hash_func)
        hash_table = {i: hash_func(data[i]) for i in range(n)}
        hash_tables.append(hash_table)

    bucket_indices = [[] for _ in range(m)]

    for i, (hash_func, hash_table) in enumerate(zip(hash_functions, hash_tables)):
        bucket_q = hash_func(q)
        bucket_range = range(bucket_q - int(R*W/2), bucket_q + int(R*W/2) + 1)

        for bucket in bucket_range:
            for idx, hash_value in hash_table.items():
                if hash_value == bucket:
                    bucket_indices[i].append(idx)

    # print sth
    collision_lengths = [len(indices) for indices in bucket_indices]
    #print(f"Generated {m} collision indices with lengths: {collision_lengths}")

    return bucket_indices

def c2lsh(data, q, config, rehash=False):
    n, d = data.shape
    k = config.NEAREST_K
    beta = config.BETA_FP
    c = config.C
    r_LSH = config.R_LSH
    
    # Calculate parameters inside the function
    p1 = p_s(config.R, config.W)
    p2 = p_s(c * config.R, config.W)
    config.ALPHA = sum([p1, p2]) / 2
    m = calculate_m(p1, p2, config)
    l = int(config.ALPHA * m)
    
    start_time = time.time()
    all_candidates = set()
    iteration = 0
    max_iterations = 20 if rehash else 1  # Set max_iterations based on rehash parameter
    compute_time = 0
    Search_time = 0

    while iteration < max_iterations:
        start = time.time()
        iteration += 1
        bucket_indices = generate_bucket_indices(data, q, m, r_LSH, config)
        collision_counts = {i: 0 for i in range(n)}
        end = time.time()

        compute_time += end - start

        start = time.time()
        # Count collisions for each data point
        for indices in bucket_indices:
            for idx in indices:
                collision_counts[idx] += 1

        # Select candidates with collision count >= l
        candidates = [(idx, count) for idx, count in collision_counts.items() if count >= l]
        candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by count in descending order

        # Add new candidates
        for idx, count in candidates:
            if idx not in all_candidates:
                all_candidates.add(idx)

        end = time.time()
        Search_time += end - start            

        if not rehash:
            break  # If not rehashing, we're done after one iteration
        elif len(all_candidates) >= k + beta * n:
            break  # If rehashing, stop when we have enough candidates
        else:
            r_LSH *= c

        if iteration == max_iterations:
            print("Reached maximum iterations without finding enough candidates.")

    # Calculate actual distances for all candidates
    candidate_distances = [(idx, euclidean(data[idx], q)) for idx in all_candidates]
    
    # Sort candidates by distance
    sorted_candidates = sorted(candidate_distances, key=lambda x: x[1])

    # Take the k nearest points
    result = sorted_candidates[:k]

    end_time = time.time()
    IO_time = end_time - start_time 

    return result, r_LSH, IO_time, Search_time

def c2lsh_with_heuristic(data, q, config, use_heuristic1=False, use_heuristic2=False, use_heuristic3=False):
    n, d = data.shape
    r_LSH = config.R_LSH
    preprocessing_time = 0
    IO_time = 0
    Search_time = 0
    start_time = time.time()

    p1 = p_s(config.R, config.W)
    p2 = p_s(config.C * config.R, config.W)
    config.ALPHA = sum([p1, p2]) / 2
    m = calculate_m(p1, p2, config)
    l = int(config.ALPHA * m)

    # Generate bucket indices
    search_start = time.time()
    collision_indices = generate_bucket_indices(data, q, m, r_LSH, config)
    search_end = time.time()
    IO_time += search_end - search_start

    # Build dyadic trees
    dyadic_start = time.time()
    dyadic_trees = [DyadicTree(n, indices, config) for indices in collision_indices]
    dyadic_end = time.time()
    preprocessing_time += dyadic_end - dyadic_start

    # Merge dyadic trees
    search_start = time.time()
    candidates = merge_dyadic_trees(dyadic_trees, p2, m, config,
                                    use_heuristic1=use_heuristic1, 
                                    use_heuristic2=use_heuristic2, 
                                    use_heuristic3=use_heuristic3)
    search_end = time.time()
    Search_time += search_end - search_start

    # Calculate actual distances for all candidates
    candidate_distances = [(idx, euclidean(data[idx], q)) for idx, _ in candidates]
    
    # Sort by distance
    sorted_candidates = sorted(candidate_distances, key=lambda x: x[1])

    # Take the k nearest points
    result = sorted_candidates[:config.NEAREST_K]

    end_time = time.time()
    total_time = end_time - start_time
    IO_time += Search_time

    return result, config.R, IO_time, preprocessing_time, Search_time, total_time





def calculate_query_accuracy(candidates, true_nn, q, k):
    if not candidates:
        return [], float('inf')  # Return empty list and infinity if no candidates found

    candidate_distances = [euclidean(c, q) for c in candidates]
    true_distances = [euclidean(t, q) for t in true_nn]
    
    # Calculate ratios only for the number of candidates actually found
    num_candidates = min(len(candidates), len(true_distances))
    ratios = [candidate_distances[i] / true_distances[i] for i in range(num_candidates)]
    
    # Calculate the average ratio based on the actual number of candidates found
    overall_ratio = np.mean(ratios) if ratios else float('inf')
    
    return ratios, overall_ratio

def get_true_knn(data, q, k):
    distances = [euclidean(point, q) for point in data]
    sorted_indices = np.argsort(distances)
    return [data[i] for i in sorted_indices[:k]]

def generate_custom_dataset(N, d, R, c, good_ratio, bad_ratio):
    print(f"Generating custom dataset: N={N}, d={d}, R={R}, c={c}")
    
    query = np.random.randn(d)
    query /= np.linalg.norm(query)
    
    data = []
    good_count = 0
    bad_count = 0

    while len(data) < N:
        point = np.random.randn(d)
        point /= np.linalg.norm(point)
        distance = np.random.uniform(0, c*R)
        point = query + point * distance
        
        dist_to_query = np.linalg.norm(point - query)
        
        if good_count < N * good_ratio:
            # Scale to be within R with some randomness
            scale_factor = np.random.uniform(0.5, 1) * (R / dist_to_query)
            point = query + (point - query) * scale_factor
            data.append(point)
            good_count += 1
        elif bad_count < N * bad_ratio:
            # Scale to be beyond c*R with some randomness
            scale_factor = np.random.uniform(1, 1.2) * (c*R / dist_to_query)
            point = query + (point - query) * scale_factor
            data.append(point)
            bad_count += 1
        else:
            # For middle range, add some randomness
            scale_factor = np.random.uniform(R/dist_to_query, c*R/dist_to_query)
            point = query + (point - query) * scale_factor
            data.append(point)
    
    data = np.array(data)
    print(f"Dataset generation completed. Total points: {len(data)}")
    return data, query

def run_algorithms_and_compute_averages(config, num_queries):
    results = {
        "Original C2LSH": {"IO_time": [], "search_time": [], "overall_ratio": []},
        "Heuristic 1": {"total_time": [], "IO_time": [], "preprocessing_time": [], "search_time": [], "overall_ratio": []},
        "Heuristic 2": {"total_time": [], "IO_time": [], "preprocessing_time": [], "search_time": [], "overall_ratio": []},
        "Heuristic 3": {"total_time": [], "IO_time": [], "preprocessing_time": [], "search_time": [], "overall_ratio": []}
    }
    
    for _ in range(num_queries):
        data, q = generate_custom_dataset(config.N, config.D, config.R, config.C, config.RATE_GOOD_PTS, config.RATE_BAD_PTS)
        true_nn = get_true_knn(data, q, config.NEAREST_K)
        
        # Original C2LSH
        result, _, IO_time, search_time = c2lsh(data, q, config, rehash=False)
        _, overall_ratio = calculate_query_accuracy([data[idx] for idx, _ in result], true_nn, q, config.NEAREST_K)
        results["Original C2LSH"]["IO_time"].append(IO_time)
        results["Original C2LSH"]["search_time"].append(search_time)
        results["Original C2LSH"]["overall_ratio"].append(overall_ratio)
        
        # C2LSH with heuristics
        heuristics = [
            ("Heuristic 1", {"use_heuristic1": True}),
            ("Heuristic 2", {"use_heuristic2": True}),
            ("Heuristic 3", {"use_heuristic3": True})
        ]

        for heuristic_name, heuristic_params in heuristics:
            result, _, IO_time, preprocessing_time, search_time, total_time = c2lsh_with_heuristic(
                data, q, config, **heuristic_params
            )
            _, overall_ratio = calculate_query_accuracy([data[idx] for idx, _ in result], true_nn, q, config.NEAREST_K)
            results[heuristic_name]["total_time"].append(total_time)
            results[heuristic_name]["IO_time"].append(IO_time)
            results[heuristic_name]["preprocessing_time"].append(preprocessing_time)
            results[heuristic_name]["search_time"].append(search_time)
            results[heuristic_name]["overall_ratio"].append(overall_ratio)
    
    # Compute and print averages
    print("\nAverage Results:")
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        for metric, values in metrics.items():
            print(f"  Average {metric}: {np.mean(values):.4f}")

def precompile_jit_functions():
    print("Precompiling JIT functions...")
    
    
    node_count = 100
    ni = 1000
    p2 = 0.5
    m = 50
    t = 3
    alpha = 0.5
    rate_bad_pts = 0.1
    i = 2
    parent_count = 200
    layer = 1
    g_bias_est = 0.0
    g_std_est = 1.0
    gamma = 0.1
    beta = 0.1
    k = 0.05
    
    
    heuristic_1(node_count, ni, p2, m, t, alpha, rate_bad_pts)
    heuristic_2(node_count, ni, p2, m, i, t, alpha)
    calculate_collision_probability(p2, alpha, m, i)
    calculate_bad_sum(p2, alpha, m, rate_bad_pts, ni, t)
    calculate_St(p2, alpha, m, ni, t, i)
    _heuristic_3_core(node_count, parent_count, m, layer, alpha, g_bias_est, g_std_est, gamma, beta, k)
    
    print("JIT functions precompiled.")

def run_algorithm_and_compute_performance(config, num_queries, use_heuristic1=False, use_heuristic2=False, use_heuristic3=False):
    total_io_time = 0
    total_search_time = 0
    total_ratio = 0

    for i in range(num_queries):
        data, q = generate_custom_dataset(config.N, config.D, config.R, config.C, config.RATE_GOOD_PTS, config.RATE_BAD_PTS)
        result, _, io_time, _, search_time, _ = c2lsh_with_heuristic(data, q, config, 
                                                                     use_heuristic1=use_heuristic1, 
                                                                     use_heuristic2=use_heuristic2, 
                                                                     use_heuristic3=use_heuristic3)

        total_io_time += io_time
        total_search_time += search_time

        true_nn = get_true_knn(data, q, config.NEAREST_K)
        candidates = [data[idx] for idx, _ in result]
        _, overall_ratio = calculate_query_accuracy(candidates, true_nn, q, config.NEAREST_K)
        total_ratio += overall_ratio

        print(f"  Query {i+1}: IO time={io_time:.2f}s, Search time={search_time:.2f}s, ratio={overall_ratio:.4f}")

    avg_io_time = total_io_time / num_queries
    avg_search_time = total_search_time / num_queries
    avg_ratio = total_ratio / num_queries

    # We want short IO and search times, and ratio close to 1
    performance = (avg_io_time + avg_search_time) * avg_ratio

    return performance, avg_io_time, avg_search_time, avg_ratio


def run_tests_for_dimension(config, num_queries):
    results = []
    
    for _ in range(num_queries):
        # Generate dataset and query point
        data, q = generate_custom_dataset(config.N, config.D, config.R, config.C, config.RATE_GOOD_PTS, config.RATE_BAD_PTS)
        true_nn = get_true_knn(data, q, config.NEAREST_K)
        
        # Test C2LSH (No Heuristics)
        result_c2lsh, _, io_time_c2lsh, search_time_c2lsh = c2lsh(data, q, config)
        ratios_c2lsh, overall_ratio_c2lsh = calculate_query_accuracy([data[idx] for idx, _ in result_c2lsh], true_nn, q, config.NEAREST_K)
        
        # Test C2LSH with Heuristics
        heuristics = [
            ("C2LSH with Heuristic 1", True, False, False),
            ("C2LSH with Heuristic 2", False, True, False),
            ("C2LSH with Heuristic 3", False, False, True)
        ]
        
        for method_name, use_h1, use_h2, use_h3 in heuristics:
            result_h, _, io_time_h, preprocessing_time_h, search_time_h, total_time_h = c2lsh_with_heuristic(
                data, q, config, use_heuristic1=use_h1, use_heuristic2=use_h2, use_heuristic3=use_h3
            )
            ratios_h, overall_ratio_h = calculate_query_accuracy([data[idx] for idx, _ in result_h], true_nn, q, config.NEAREST_K)
            
            results.append({
                'method': method_name,
                'total_time': total_time_h,
                'io_time': io_time_h,
                'search_time': search_time_h,
                'preprocessing_time': preprocessing_time_h,
                'overall_ratio': overall_ratio_h,
                'candidates': len(result_h),
                'top_5_ratios': ratios_h[:5]
            })
        
        # Add C2LSH results
        results.append({
            'method': "C2LSH (No Heuristics)",
            'total_time': io_time_c2lsh + search_time_c2lsh,
            'io_time': io_time_c2lsh,
            'search_time': search_time_c2lsh,
            'preprocessing_time': 0,
            'overall_ratio': overall_ratio_c2lsh,
            'candidates': len(result_c2lsh),
            'top_5_ratios': ratios_c2lsh[:5]
        })
    
    # Calculate averages
    avg_results = {}
    for key in results[0].keys():
        if key == 'method':
            continue
        for method in set(r['method'] for r in results):
            method_results = [r[key] for r in results if r['method'] == method]
            if key == 'top_5_ratios':
                avg_results.setdefault(method, {})[key] = [sum(x)/len(x) if x else float('inf') for x in zip(*method_results)]
            else:
                avg_results.setdefault(method, {})[key] = sum(method_results) / len(method_results)
    
    return avg_results

def print_results(results):
    for method, result in results.items():
        print(f"\nResults for {method}:")
        print(f"Avg Total Time: {result['total_time']:.4f}s")
        print(f"Avg IO Time: {result['io_time']:.4f}s")
        print(f"Avg Search Time: {result['search_time']:.4f}s")
        if 'preprocessing_time' in result:
            print(f"Avg Preprocessing Time: {result['preprocessing_time']:.4f}s")
        
        if result['overall_ratio'] == float('inf'):
            print("No candidates found.")
        else:
            print(f"Avg Overall Ratio: {result['overall_ratio']:.4f}")
            print(f"Avg Candidates: {result['candidates']:.2f}")
            print("Avg Ratios for top 5 candidates:")
            for i, ratio in enumerate(result['top_5_ratios']):
                if ratio == float('inf'):
                    print(f"  Candidate {i+1}: N/A")
                else:
                    print(f"  Candidate {i+1}: {ratio:.4f}")

def test_c2lsh_and_heuristics():
    config = Config()
    N = 10000  # Fixed number of points
    d_values = [10]  # Different dimensions to test
    num_queries = 3  # Number of test queries

    all_results = []

    for d in d_values:
        print(f"\nTesting with d = {d}")
        config.N = N
        config.D = d

        # Run tests for current dimension
        results = run_tests_for_dimension(config, num_queries)
        
        # Store results
        for method, result in results.items():
            result['method'] = method
            result['N'] = N
            result['d'] = d
            all_results.append(result)
        
        # Print results for current dimension
        print_results(results)

    # Print summary of all tests
    print("\nSummary of all tests:")
    for result in all_results:
        print(f"d = {result['d']}, Method: {result['method']}")
        print(f"  Avg Total Time: {result['total_time']:.4f}s")
        print(f"  Avg IO Time: {result['io_time']:.4f}s")
        print(f"  Avg Search Time: {result['search_time']:.4f}s")
        if 'preprocessing_time' in result:
            print(f"  Avg Preprocessing Time: {result['preprocessing_time']:.4f}s")
        
        if result['overall_ratio'] == float('inf'):
            print("  No candidates found.")
        else:
            print(f"  Avg Overall Ratio: {result['overall_ratio']:.4f}")
            print(f"  Avg Candidates: {result['candidates']:.2f}")
            print("  Avg Ratios for top 5 candidates:")
            for i, ratio in enumerate(result['top_5_ratios']):
                if ratio == float('inf'):
                    print(f"    Candidate {i+1}: N/A")
                else:
                    print(f"    Candidate {i+1}: {ratio:.4f}")
        print()


class Config:
    def __init__(self):
        self.N = 40000
        self.BETA_FP = 100/self.N
        self.SIGMA = 0.05
        self.ALPHA = 0.5
        self.R = 1
        self.C = 3
        self.W = 1.0
        self.D = 1000

        self.RATE_GOOD_PTS = 0.02
        self.RATE_BAD_PTS = 0.6

        self.T_INIT = 16
        

        self.G_BIAS_EST = 0
        self.G_STD_EST = 1
        self.GAMMA = 0.15
        self.BETA = 0.1
        self.K = 0.01

        self.R_LSH = 1.0
        self.NEAREST_K = 10






def compare_algorithms(config):
    RATE_GOOD_PTS_values = [0.001, 0.005, 0.01 ,0.05, 0.1 ,0.3]
    num_queries = 10  # Run 3 queries for each configuration and take the average
    algorithms = ["C2LSH******", "Heuristic 1", "Heuristic 2", "Heuristic 3"]
    
    results = {}
    
    for rate in RATE_GOOD_PTS_values:
        config.RATE_GOOD_PTS = rate
        
        print(f"\nExperiment for RATE_GOOD_PTS={rate}:")
        
        for algo in algorithms:
            total_io_time = 0
            total_search_time = 0
            total_preprocessing_time = 0
            total_ratio = 0
            total_candidates = 0
            valid_ratio_count = 0
            individual_ratios = [0] * config.NEAREST_K
            
            print(f"\n  Running {algo}:")
            for query_num in range(num_queries):
                print(f"    Query {query_num + 1}:")
                data, q = generate_custom_dataset(config.N, config.D, config.R, config.C, config.RATE_GOOD_PTS, config.RATE_BAD_PTS)
                
                if algo == "C2LSH******":
                    result, _, io_time, search_time = c2lsh(data, q, config)
                    preprocessing_time = 0
                else:
                    result, _, io_time, preprocessing_time, search_time, _ = c2lsh_with_heuristic(
                        data, q, config, 
                        use_heuristic1=(algo == "Heuristic 1"),
                        use_heuristic2=(algo == "Heuristic 2"),
                        use_heuristic3=(algo == "Heuristic 3")
                    )
                
                true_nn = get_true_knn(data, q, config.NEAREST_K)
                candidates = [data[idx] for idx, _ in result]
                ratios, overall_ratio = calculate_query_accuracy(candidates, true_nn, q, config.NEAREST_K)
                
                print(f"      IO time: {io_time:.4f}s")
                print(f"      Search time: {search_time:.4f}s")
                print(f"      Preprocessing time: {preprocessing_time:.4f}s")
                print(f"      Number of candidates found: {len(candidates)}")
                print(f"      Individual ratios: {ratios}")
                print(f"      Overall ratio: {overall_ratio}")
                
                total_io_time += io_time
                total_search_time += search_time
                total_preprocessing_time += preprocessing_time
                if overall_ratio != float('inf'):
                    total_ratio += overall_ratio
                    valid_ratio_count += 1
                total_candidates += len(candidates)
                
                for i, ratio in enumerate(ratios):
                    if ratio != float('inf'):
                        individual_ratios[i] += ratio
            
            avg_io_time = total_io_time / num_queries
            avg_search_time = total_search_time / num_queries
            avg_preprocessing_time = total_preprocessing_time / num_queries
            avg_ratio = total_ratio / valid_ratio_count if valid_ratio_count > 0 else float('inf')
            avg_candidates = total_candidates / num_queries
            avg_individual_ratios = [r / valid_ratio_count if valid_ratio_count > 0 else float('inf') for r in individual_ratios]
            
            results[(rate, algo)] = {
                'avg_io_time': avg_io_time,
                'avg_search_time': avg_search_time,
                'avg_preprocessing_time': avg_preprocessing_time,
                'avg_ratio': avg_ratio,
                'avg_candidates': avg_candidates,
                'valid_ratio_count': valid_ratio_count,
                'avg_individual_ratios': avg_individual_ratios
            }
            
            print(f"\n  Summary for {algo}:")
            print(f"    Avg IO time: {avg_io_time:.4f}s")
            print(f"    Avg Search time: {avg_search_time:.4f}s")
            print(f"    Avg Preprocessing time: {avg_preprocessing_time:.4f}s")
            print(f"    Avg Ratio: {avg_ratio:.4f}")
            print(f"    Avg Candidates: {avg_candidates:.2f}")
            print(f"    Valid ratio count: {valid_ratio_count}/{num_queries}")
            print(f"    Avg Individual ratios: {avg_individual_ratios}")
    
    return results

def print_comparison_results(results):
    print("\nComparison Results:")
    for rate in sorted(set(rate for rate, _ in results.keys())):
        print(f"\nResults for RATE_GOOD_PTS={rate}:")
        print("Algorithm\tIO Time\tSearch Time\tPreprocessing Time\tRatio\tAvg Candidates\tValid Ratios\tIndividual Ratios")
        for algo in ["C2LSH******", "Heuristic 1", "Heuristic 2", "Heuristic 3"]:
            metrics = results[(rate, algo)]
            individual_ratios_str = ', '.join([f'{r:.4f}' for r in metrics['avg_individual_ratios']])
            print(f"{algo}\t{metrics['avg_io_time']:.4f}\t{metrics['avg_search_time']:.4f}\t{metrics['avg_preprocessing_time']:.4f}\t{metrics['avg_ratio']:.4f}\t{metrics['avg_candidates']:.2f}\t{metrics['valid_ratio_count']}/3\t[{individual_ratios_str}]")

# Main function
if __name__ == "__main__":
    print("Starting comparison experiment")
    config = Config()
    precompile_jit_functions()
    results = compare_algorithms(config)
    print_comparison_results(results)
    print("Experiment completed")
