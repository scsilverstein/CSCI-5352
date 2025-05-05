#!/usr/bin/env python3

import graph_tool.all as gt
import pandas as pd
import numpy as np
import load_data
from tqdm.auto import tqdm
from build_graph import build_graph
from graph_drawing import draw_flat_sbm, draw_nested_sbm_levels, plot_block_metrics, draw_nested_sbm, draw_block_matrix_heatmap
import os
import pickle 

def load_cache(cache_path):
    print(f"Attempting to load cached data from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print("Successfully loaded cached data.")
    return data


def draw_and_plot_data_set(data_set):
    name = data_set['name']
    print(f"\n{'='*10} Drawing and Plotting Dataset: {name} {'='*10}")

    print(f"\n--- Building Graph for {name} ---")
    name_check, graph = build_graph(data_set)
    print(f"Graph built: {graph.num_vertices()} vertices, {graph.num_edges()} edges.")
    print(f"Graph summary: directed={graph.is_directed()}, weighted={'weight' in graph.ep}")

    print(f"\n--- Loading Cached SBM Results for {name} ---")
    CACHE_DIR = "./cache"
    graph_id = f"{graph.num_vertices()}_{graph.num_edges()}"
    cache_file = os.path.join(CACHE_DIR, f"{name}_{graph_id}_sbm_results.pkl")

    sbm_results = load_cache(cache_file)

    _, _, _, flat_exponential_result, flat_lognormal_result, nested_exponential_result, nested_lognormal_result = sbm_results
    print(f"--- SBM Results Loaded for {name} ---")

    flat_exp_state = flat_exponential_result[0]
    flat_lognorm_state = flat_lognormal_result[0]
    nested_exp_state = nested_exponential_result[0]
    nested_lognorm_state = nested_lognormal_result[0]

    print(f"\n--- Drawing Graphs for {name} ---")
    draw_flat_sbm(flat_exp_state, graph, f"{name}_flat_exponential")
    draw_flat_sbm(flat_lognorm_state, graph, f"{name}_flat_lognormal")

    draw_nested_sbm(nested_exp_state, graph, f"{name}_nested_exponential_hierarchical")
    draw_nested_sbm_levels(nested_exp_state, graph, f"{name}_nested_exponential")
    draw_nested_sbm(nested_lognorm_state, graph, f"{name}_nested_lognormal_hierarchical")
    draw_nested_sbm_levels(nested_lognorm_state, graph, f"{name}_nested_lognormal")

    draw_block_matrix_heatmap(flat_exp_state, graph, f"{name}_flat_exponential", matrix_type='parameters', rec_type='real-exponential')
    draw_block_matrix_heatmap(flat_lognorm_state, graph, f"{name}_flat_lognormal", matrix_type='parameters', rec_type='real-normal')
    draw_block_matrix_heatmap(nested_exp_state.get_levels()[0], graph, f"{name}_nested_exponential_l0", matrix_type='parameters', rec_type='real-exponential')
    draw_block_matrix_heatmap(nested_lognorm_state.get_levels()[0], graph, f"{name}_nested_lognormal_l0", matrix_type='parameters',rec_type='real-normal')

    print(f"--- Graph Drawing Complete for {name} ---")

    print(f"Generating block metric plots for {name}...")
    summary_filename = f"{name}_block_summary_metrics.csv"
    summary_df = pd.read_csv(summary_filename)
    plot_block_metrics(summary_df, name)

    print(f"\n{'='*10} Finished Drawing and Plotting Dataset: {name} {'='*10}")


if __name__ == "__main__":
    print("Starting Graph Drawing and Plotting Pipeline...")

    print("\n--- Loading Data ---")
    data_sets = load_data.main()
    print(f"Loaded {len(data_sets)} datasets.")

    print(f"\n--- Processing {len(data_sets)} Datasets for Drawing and Plotting ---")
    for data_set in tqdm(data_sets, desc="Processing Datasets"):
        draw_and_plot_data_set(data_set)

    print("\nGraph Drawing and Plotting Pipeline Finished.")