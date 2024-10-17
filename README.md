# MAST90109-Dyadic-C2LSH
# Here is the code for Dyadic C2LSH Algorithm
# CODE is all in HA_sim.py
## Introduction
This project implements a variation of the **C2LSH (Collision Counting Locality Sensitive Hashing)** algorithm for efficient nearest neighbor search. It incorporates heuristics to optimize the merging of **dyadic trees** based on collision data. The project includes functions for dataset generation, collision counting, dyadic tree construction, and heuristic-based optimizations.

## Table of Contents
- [Node Class](#1-node-class)
- [DyadicTree Class](#2-dyadictree-class)
- [merge_dyadic_trees Function](#3-mergedyadictrees-function)
- [Heuristics](#4-heuristics-heuristic_1-heuristic_2-heuristic_3)
- [generate_bucket_indices Function](#5-generate_bucket_indices-function)
- [c2lsh Function](#6-c2lsh-function)
- [c2lsh_with_heuristic Function](#7-c2lsh_with_heuristic-function)
- [calculate_query_accuracy Function](#8-calculate_query_accuracy-function)
- [generate_custom_dataset Function](#9-generate_custom_dataset-function)
- [run_algorithms_and_compute_averages Function](#10-run_algorithms_and_compute_averages-function)
- [compare_algorithms Function](#11-compare_algorithms-and-print_comparison_results-functions)
- [Config Class](#12-config-class)

## 1. `Node` Class
Represents a node in a dyadic tree:
- Stores information like collision count, node index, number of points (`ni`), and tree layer.
- Links parent and child nodes to create a hierarchical tree structure.

## 2. `DyadicTree` Class
This class constructs a dyadic tree from collision data:
- **`build_tree`**: Initializes bottom-layer nodes and builds upward by merging nodes into parent nodes.
- **`print_tree`**: Prints the structure of the tree for debugging.
- **`find_node`**: Helps locate specific nodes in the tree using layer and index.

## 3. `merge_dyadic_trees` Function
Merges multiple dyadic trees to form a candidate set:
- Applies three heuristics (`heuristic_1`, `heuristic_2`, `heuristic_3`) to filter nodes during the merging process.
- Uses a priority queue (heap) to manage node processing.
- Returns a set of candidates after merging.

## 4. Heuristics (`heuristic_1`, `heuristic_2`, `heuristic_3`)
- **`heuristic_1`**: Determines if a node’s count satisfies conditions based on the expected number of collisions and bad points.
- **`heuristic_2`**: Uses collision probability to decide if a node should be included in the candidate set.
- **`heuristic_3`**: Checks for deviations between a node's count and its parent's count to identify important nodes.

## 5. `generate_bucket_indices` Function
Generates hash buckets for the data points:
- Hashes the dataset into different buckets based on random projections.
- For each query point, it identifies collision points based on hash bucket proximity.

## 6. `c2lsh` Function
Core implementation of the **C2LSH algorithm**:
- Hashes the data points and counts collisions in each bucket.
- Selects candidate points that exceed a collision threshold.
- Returns the k-nearest points to the query based on Euclidean distance.

## 7. `c2lsh_with_heuristic` Function
An optimized version of `c2lsh` that integrates dyadic tree structures:
- Builds dyadic trees from collision data and uses **heuristics** for optimized candidate selection.
- Tracks time spent on IO, preprocessing, and search.
- Returns the k-nearest neighbors based on Euclidean distance.

## 8. `calculate_query_accuracy` Function
Computes the accuracy of the results by comparing the distances of retrieved candidates to the true nearest neighbors:
- Returns the ratio of distances and the overall accuracy.

## 9. `generate_custom_dataset` Function
Generates a synthetic dataset with a mixture of:
- "Good points" (close to the query) and 
- "Bad points" (far from the query).
Used to test and benchmark the C2LSH algorithm and its heuristics.

## 10. `run_algorithms_and_compute_averages` Function
Runs multiple queries using different variations of the C2LSH algorithm:
- Computes and prints the average IO time, search time, and accuracy for each method.

## 11. `compare_algorithms` and `print_comparison_results` Functions
Compares the performance of different algorithms over multiple queries:
- Prints the results in a detailed manner to compare IO times, search times, and accuracy ratios.

## 12. `Config` Class
Holds configuration parameters used throughout the project, such as:
- Dataset size (`N`), collision thresholds, search radius (`R`), number of nearest neighbors (`NEAREST_K`), and heuristic parameters (`ALPHA`, `BETA_FP`).
  
## Execution Steps
1. **Precompile JIT Functions**: Run `precompile_jit_functions()` to compile functions with Numba for faster execution.
2. **Run Experiments**: Use `compare_algorithms()` to test and compare various versions of C2LSH.
3. **Performance Evaluation**: Use `print_comparison_results()` to display the performance of each algorithm based on time and accuracy.
