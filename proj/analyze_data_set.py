import graph_tool.all as gt
import pandas as pd
import numpy as np
import os
import pickle
from tqdm.auto import tqdm

from build_graph import build_graph
from sbm_analysis import minimize_graph
from block_analysis import analyze_sbm_blocks
from analyze_blocks import analyze_blocks 
from graph_drawing import draw_flat_sbm, draw_nested_sbm_levels, draw_nested_sbm, draw_block_matrix_heatmap, draw_hierarchical_visualization, plot_weight_distribution_fit
from reporting import generate_summary_report


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


def preprocess_graph(graph):
    print("Preprocessing graph (normalizing weights)...")
    weights = graph.ep.weight.fa

    for e in graph.edges():
         w = graph.ep.weight[e]
         graph.ep.weight[e] = np.log1p(w)
    print("Applied log1p transformation to edge weights.")

    return graph


def analyze_data_set(data_set):
    name = data_set['name']
    print(f"\n{'='*10} Analyzing Dataset: {name} {'='*10}")

    print(f"\n--- Building Graph for {name} ---")
    name_check, graph = build_graph(data_set)
    print(f"Graph built: {graph.num_vertices()} vertices, {graph.num_edges()} edges.")
    print(f"Graph summary: directed={graph.is_directed()}, weighted={'weight' in graph.ep}")

    graph = preprocess_graph(graph)

    print(f"\n--- Running SBM Analysis for {name} ---")
    CACHE_DIR = "./cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    graph_id = f"{graph.num_vertices()}_{graph.num_edges()}"
    cache_file = os.path.join(CACHE_DIR, f"{name}_{graph_id}_sbm_results.pkl")

    sbm_results = load_or_compute(cache_file, minimize_graph, graph)

    best_state, best_dl, best_type, flat_exponential_result, flat_lognormal_result, flat_pp_result, nested_exponential_result, nested_lognormal_result = sbm_results
    print(f"--- SBM Analysis Complete for {name} ---")
    print(f"--- Best Model: {best_type}, DL: {best_dl} ---")

    flat_blockmodel = None
    nested_blockmodel = None

    if best_type == "flat_exponential":
        flat_blockmodel = flat_exponential_result[0]
        print("  Best model is Flat Exponential.")
    elif best_type == "flat_lognormal":
        flat_blockmodel = flat_lognormal_result[0]
        print("  Best model is Flat Log-Normal.")
    elif best_type == "flat_pp":
        flat_blockmodel = flat_pp_result[0]
        print("  Best model is Flat PP (Unweighted).")
    elif best_type == "nested_exponential":
        nested_blockmodel = nested_exponential_result[0]
        flat_blockmodel = nested_blockmodel.project_level(0)
        print("  Best model is Nested Exponential. Derived flat blockmodel from level 0.")
    elif best_type == "nested_lognormal":
        nested_blockmodel = nested_lognormal_result[0]
        flat_blockmodel = nested_blockmodel.project_level(0)
        print("  Best model is Nested Log-Normal. Derived flat blockmodel from level 0.")
    else:
        print("  No best model determined or fitted successfully.")


    print(f"\n--- Analyzing Metrics within SBM Blocks for {name} ---")
    graph_id = f"{graph.num_vertices()}_{graph.num_edges()}"
    analysis_cache_file = os.path.join(CACHE_DIR, f"{name}_{graph_id}_analysis_results.pkl")
    analysis_results = load_or_compute(
        analysis_cache_file,
        analyze_sbm_blocks,
        graph,
        flat_blockmodel,
        nested_blockmodel,
        flat_exponential_result=flat_exponential_result,
        flat_lognormal_result=flat_lognormal_result,
        flat_pp_result=flat_pp_result,
        nested_exponential_result=nested_exponential_result,
        nested_lognormal_result=nested_lognormal_result
    )

    weighted_modularity, weighted_assortativity = analyze_blocks(graph, best_state, best_type)

    summary_df, centrality_df, shortest_distance_df, inter_block_metrics_df, nested_block_summaries_by_level, nested_node_centralities_by_level = analysis_results

    print('Saving Results', summary_df.shape, centrality_df.shape, shortest_distance_df.shape, inter_block_metrics_df.shape)
    summary_filename = f"{name}_block_summary_metrics.csv"
    centrality_filename = f"{name}_node_centrality_metrics.csv"
    shortest_distance_filename = f"{name}_weighted_shortest_distances.csv"
    inter_block_metrics_filename = f"{name}_inter_block_metrics.csv"

    save_df(summary_df, summary_filename, "Block summary metrics", name)
    save_df(centrality_df, centrality_filename, "Node centrality metrics", name)
    save_df(shortest_distance_df, shortest_distance_filename, "Weighted shortest distances", name)
    save_df(inter_block_metrics_df, inter_block_metrics_filename, "Inter-block metrics", name)

    print(f"\n--- Calculating Global Metrics for {name} ---")
    global_metrics = {}
    print("Calculating density...")
    num_vertices = graph.num_vertices()
    num_edges = graph.num_edges()
    max_possible_edges = num_vertices * (num_vertices - 1)
    global_metrics['density'] = num_edges / max_possible_edges

    print("Calculating global clustering coefficient...")
    clustering_key = 'global_clustering_coefficient'
    unweighted_clustering_key ='global_unweighted_clustering_coefficient'
    clustering_result = gt.global_clustering(graph, weight=graph.ep.weight)
    unweighted_clustering_result = gt.global_clustering(graph)
    global_metrics[clustering_key] = clustering_result[0]
    global_metrics[unweighted_clustering_key] = unweighted_clustering_result[0]
    print(f"Weighted global clustering coefficient calculated: {global_metrics.get(clustering_key, 'N/A')}")
    print("... Global clustering finished.")

    print("Calculating reciprocity...")
    reciprocity_key = 'reciprocity'
    weighted_reciprocity = 'weighted_reciprocity'
    reciprocity_value = gt.edge_reciprocity(graph)
    weighted_reciprocity_value = gt.edge_reciprocity(graph,weight=graph.ep.weight)
    global_metrics[reciprocity_key] = reciprocity_value
    global_metrics[weighted_reciprocity] = weighted_reciprocity_value
    print(f"Reciprocity calculated: {reciprocity_value}")

    print("Calculating degree assortativity...")
    assort_key = 'degree_assortativity'
    weighted_assort_key = 'weighted_assortativity'
    assort_val, _ = gt.assortativity(graph, "total")
    weighted_assort_val, _ = gt.assortativity(graph, "total", eweight=graph.ep.weight)
    global_metrics[assort_key] = assort_val
    global_metrics[weighted_assort_key] = weighted_assort_val
    print(f"Standard (Unweighted) degree assortativity calculated: {global_metrics.get(assort_key, 'N/A')}")
    print(f"Standard (Weighted) degree assortativity calculated: {global_metrics.get(weighted_assort_key, 'N/A')}")
    print("... Degree assortativity finished.")

    print("Calculating average weighted shortest path length (this may take time)...")
    shortest_path_key = 'avg_weighted_shortest_path'
    dist_map = gt.shortest_distance(graph, weights=graph.ep.weight)
    all_distances = []
    for v in graph.vertices():
        distances_from_v = dist_map[v].a
        finite_dists = distances_from_v[np.isfinite(distances_from_v)]
        all_distances.extend(finite_dists)

    mean_dist = np.mean(all_distances, dtype=np.longdouble)
    global_metrics[shortest_path_key] = mean_dist
    print(f"Average weighted shortest path calculated: {global_metrics.get(shortest_path_key, 'N/A')}")
    print("... Average weighted shortest path finished.")

    print(f"Global Metrics Calculated: {global_metrics}")


    print(f"\n--- Generating Summary Report for {name} ---")
    report_content = generate_summary_report(
        name=name,
        graph=graph,
        best_state=best_state,
        summary_df=summary_df,
        centrality_df=centrality_df,
        shortest_distance_df=shortest_distance_df,
        inter_block_metrics_df=inter_block_metrics_df,
        global_metrics=global_metrics,
        weighted_modularity=weighted_modularity,
        weighted_assortativity=weighted_assortativity,
        
        nested_block_summaries_by_level=nested_block_summaries_by_level,
        nested_node_centralities_by_level=nested_node_centralities_by_level,
        flat_exponential_result=flat_exponential_result,
        flat_lognormal_result=flat_lognormal_result,
        flat_pp_result=flat_pp_result,
        nested_exponential_result=nested_exponential_result,
        nested_lognormal_result=nested_lognormal_result,
        best_type=best_type,
        best_dl=best_dl
    )
    report_filename = f"{name}_summary_report.txt"
    with open(report_filename, 'w') as f:
        f.write(report_content)
    print(f"Summary report saved to {report_filename}")


    print(f"\n--- Drawing Graphs for {name} ---")
    flat_exp_state = flat_exponential_result[0]
    flat_lognorm_state = flat_lognormal_result[0]
    flat_pp_state = flat_pp_result[0]
    nested_exp_state = nested_exponential_result[0]
    nested_lognorm_state = nested_lognormal_result[0]

    draw_flat_sbm(flat_exp_state, graph, f"{name}_flat_exponential")
    draw_flat_sbm(flat_lognorm_state, graph, f"{name}_flat_lognormal")
    draw_flat_sbm(flat_pp_state, graph, f"{name}_flat_pp")

    draw_nested_sbm(nested_exp_state, graph, f"{name}_nested_exponential_hierarchical")
    draw_nested_sbm_levels(nested_exp_state, graph, f"{name}_nested_exponential")
    draw_hierarchical_visualization(nested_exp_state, graph, f"{name}_nested_exponential_hierarchical")
    draw_nested_sbm(nested_lognorm_state, graph, f"{name}_nested_lognormal_hierarchical")
    draw_nested_sbm_levels(nested_lognorm_state, graph, f"{name}_nested_lognormal")
    draw_hierarchical_visualization(nested_lognorm_state, graph, f"{name}_nested_lognormal_hierarchical")

    draw_block_matrix_heatmap(flat_exp_state, graph, f"{name}_flat_exponential", matrix_type='parameters', rec_type='real-exponential')
    draw_block_matrix_heatmap(flat_lognorm_state, graph, f"{name}_flat_lognormal", matrix_type='parameters', rec_type='real-normal')

    draw_block_matrix_heatmap(nested_exp_state.get_levels()[0], graph, f"{name}_nested_exponential_l0", matrix_type='parameters', rec_type='real-exponential')
    draw_block_matrix_heatmap(nested_exp_state.get_levels()[1], graph, f"{name}_nested_exponential_l1", matrix_type='parameters', rec_type='real-exponential')
    draw_block_matrix_heatmap(nested_lognorm_state.get_levels()[0], graph, f"{name}_nested_lognormal_l0", matrix_type='parameters',rec_type='real-normal')
    draw_block_matrix_heatmap(nested_lognorm_state.get_levels()[1], graph, f"{name}_nested_lognormal_l1", matrix_type='parameters',rec_type='real-normal')

    print(f"  Plotting weight distribution fits for {name}...")
    # plot_weight_distribution_fit(flat_exp_state, graph, f"{name}_flat_exponential", rec_type='real-exponential')
    # plot_weight_distribution_fit(flat_lognorm_state, graph, f"{name}_flat_lognormal", rec_type='real-normal')
    # plot_weight_distribution_fit(nested_exp_state.get_levels()[0], graph, f"{name}_nested_exponential_l0", rec_type='real-exponential')
    # plot_weight_distribution_fit(nested_lognorm_state.get_levels()[0], graph, f"{name}_nested_lognormal_l0", rec_type='real-normal')

    print(f"--- Graph Drawing Complete for {name} ---")

    print(f"\n{'='*10} Finished Processing Dataset: {name} {'='*10}")

if __name__ == "__main__":
    import load_data
    print("Loading data...")
    data_sets = load_data.main()
    print(f"Loaded {len(data_sets)} datasets.")
    target_dataset = next((ds for ds in data_sets if ds.get('name') == 'country_to_country'))
    analyze_data_set(target_dataset)
