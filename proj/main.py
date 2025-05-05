#!/usr/bin/env python3

import graph_tool.all as gt # Keep for GraphView potentially needed? Check block_analysis usage
import pandas as pd
import numpy as np
import load_data
import os # Added for path operations
import pickle # Added for caching
from tqdm.auto import tqdm # Import tqdm for progress bars
from analyze_data_set import analyze_data_set # Import the analysis function
# from graph_drawing import draw_flat_sbm, draw_nested_sbm_levels, plot_block_metrics, draw_nested_sbm, draw_block_matrix_heatmap # Keep for drawing functions
# import csv # No longer needed


# --- Helper Functions ---

def load_or_compute(cache_path, compute_func, *args, **kwargs):
    if os.path.exists(cache_path):
        print(f"Attempting to load cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded cached data.")
        return data
    else:
        print(f"No cache found at {cache_path}. Computing...")

    computed_data = compute_func(*args, **kwargs)
    print(f"Computation complete for {compute_func.__name__}.")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(computed_data, f)
    print(f"Data saved to cache: {cache_path}")
    return computed_data


def save_df(df, filename, df_description, dataset_name):
    df.to_csv(filename, index=False)
    print(f"{df_description} saved to {filename}")


# --- Main Analysis Functions ---

def preprocess_graph(graph):
    print("Preprocessing graph (normalizing weights)...")
    weights = graph.ep.weight.fa

    log_weights = graph.new_edge_property("double")
    for e in graph.edges():
         w = graph.ep.weight[e]
         log_weights[e] = np.log1p(w)
    graph.ep.weight = log_weights
    print("Applied log1p transformation to edge weights.")

    # Removed commented-out edge filtering code

    return graph



# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Full Analysis Pipeline...")

    print("\n--- Loading Data ---")
    data_sets = load_data.main()
    print(f"Loaded {len(data_sets)} datasets.")

    print(f"\n--- Analyzing {len(data_sets)} Datasets ---")
    for data_set in tqdm(data_sets, desc="Analyzing Datasets"):
        analyze_data_set(data_set)

    print("\nAnalysis Pipeline Finished.")
