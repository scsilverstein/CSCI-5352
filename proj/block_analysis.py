import graph_tool.all as gt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import pickle

def _calculate_block_metrics_internal(subgraph, original_graph_vp_name, block_id, sbm_type):
    metrics = {"sbm_type": sbm_type, "block_id": block_id}
    node_centralities = []


    num_vertices = subgraph.num_vertices()
    num_edges = subgraph.num_edges()
    metrics["num_vertices"] = num_vertices
    metrics["num_edges"] = num_edges

    # Block Level
    # Density
    density = num_edges / (num_vertices * (num_vertices - 1))
    metrics["density"] = float(density)

    # Clustering
    local_clust = gt.local_clustering(subgraph, weight=subgraph.ep.weight)
    metrics["avg_local_clustering"] = float(np.mean(local_clust.a))

    # Reciprocity
    recip = gt.reciprocity(subgraph)
    metrics["reciprocity"] = float(recip)

    # Assortativity
    assort_result = gt.assortativity(subgraph, "total", eweight=subgraph.ep.weight)
    assort_value = float(assort_result[0])
    assort_type = "weighted"

    metrics["assortativity"] = assort_value
    metrics["assortativity_type"] = assort_type

    # Node level
    # Weighted Degree
    in_degree_pm = subgraph.degree_property_map("in", weight=subgraph.ep.weight)
    out_degree_pm = subgraph.degree_property_map("out", weight=subgraph.ep.weight)

    # Betweenness
    bv_pm, be_pm = gt.betweenness(subgraph, weight=subgraph.ep.weight)

    # PageRank
    pr_pm = gt.pagerank(subgraph, weight=subgraph.ep.weight)

    for v_sub in subgraph.vertices():
        v_orig = subgraph.vertex_index[v_sub] # Get original vertex index

        in_deg_val = float(in_degree_pm[v_sub])
        out_deg_val = float(out_degree_pm[v_sub])
        bv_val = float(bv_pm[v_sub])
        pr_val = float(pr_pm[v_sub])

        node_centralities.append({
            "sbm_type": sbm_type,
            "block_id": block_id,
            "vertex_name": original_graph_vp_name[v_orig],
            "in_degree": in_deg_val,
            "out_degree": out_deg_val,
            "betweenness": bv_val,
            "pagerank": pr_val
        })

    return metrics, node_centralities


from collections import defaultdict
from typing import Dict, Any

def calculate_inter_block_metrics(graph: gt.Graph, blockmodel) -> pd.DataFrame:
    blocks = blockmodel.get_blocks()

    inter_block_metrics = defaultdict(lambda: {'total_weight': 0.0, 'num_edges': 0})

    weights = graph.ep.get("weight", None)

    for edge in graph.edges():
        source_node = edge.source()
        target_node = edge.target()

        # Access block ID using the vertex descriptor directly on the PropertyMap/dict
        source_block_id = blocks[source_node]
        target_block_id = blocks[target_node]

        block_pair = (int(source_block_id), int(target_block_id))

        # Increment edge count
        inter_block_metrics[block_pair]['num_edges'] += 1

        edge_weight = float(weights[edge])
        inter_block_metrics[block_pair]['total_weight'] += edge_weight

    data_for_df = []
    for (source_block, target_block), metrics in inter_block_metrics.items():
        data_for_df.append({
            "source_block": source_block,
            "target_block": target_block,
            "total_weight": metrics['total_weight'],
            "num_edges": metrics['num_edges']
        })

    return pd.DataFrame(data_for_df)


def _process_nested_sbm_level(graph, nested_state, sbm_base_name, nested_block_summaries_by_level, nested_node_centralities_by_level, all_inter_block_metrics):
    print(f"\nProcessing {sbm_base_name.replace('_', ' ').title()} SBM Levels...")
    for level_idx, level_state in enumerate(tqdm(nested_state.get_levels(), desc=f"Analyzing {sbm_base_name.replace('_', ' ').title()} Levels")):
        print(f"  Processing Level {level_idx} of {sbm_base_name.replace('_', ' ').title()}...")
        original_graph_blocks = nested_state.project_level(level_idx).get_blocks()

        num_level_blocks = original_graph_blocks.a.max() + 1
        level_sbm_type = f"{sbm_base_name}_l{level_idx}"

        print(f"  Processing {num_level_blocks} blocks from {sbm_base_name.replace('_', ' ').title()} SBM Level {level_idx}...")

        level_block_summaries = []
        level_node_centralities = []

        for b_id in tqdm(range(num_level_blocks), desc=f"Level {level_idx} Blocks", leave=False):
            vfilt = original_graph_blocks.a == b_id
            num_nodes_in_block = np.sum(vfilt)

            subgraph = gt.GraphView(graph, vfilt=vfilt)
            block_summary, node_cents = _calculate_block_metrics_internal(subgraph, graph.vp.name, b_id, level_sbm_type)
            level_block_summaries.append(block_summary)
            level_node_centralities.extend(node_cents)


        nested_block_summaries_by_level[level_sbm_type] = pd.DataFrame(level_block_summaries)
        nested_node_centralities_by_level[level_sbm_type] = pd.DataFrame(level_node_centralities)

        level_inter_block_metrics = calculate_inter_block_metrics(graph, level_state)
        level_inter_block_metrics['sbm_type'] = level_sbm_type
        all_inter_block_metrics = pd.concat([all_inter_block_metrics, level_inter_block_metrics], ignore_index=True)


    print(f"Finished processing {sbm_base_name.replace('_', ' ').title()} SBM Levels.")
    return all_inter_block_metrics


def analyze_sbm_blocks(graph, flat_blockmodel, nested_blockmodel, flat_exponential_result, flat_lognormal_result, flat_pp_result, nested_exponential_result, nested_lognormal_result):
    cache_dir = ".sbm_cache"
    os.makedirs(cache_dir, exist_ok=True)
    graph_id = f"{graph.num_vertices()}_{graph.num_edges()}"
    cache_file = os.path.join(cache_dir, f"{graph_id}_analysis_results.pkl")

    if os.path.exists(cache_file):
        print(f"\nCache found for analysis results ({graph_id}). Loading results...")
        with open(cache_file, 'rb') as f:
            cached_results = pickle.load(f)
        print("Loaded cached analysis results.")
        return cached_results

    all_block_summaries = []
    all_node_centralities = []
    all_inter_block_metrics = pd.DataFrame()
    nested_block_summaries_by_level = {}
    nested_node_centralities_by_level = {}

    shortest_distance_df = pd.DataFrame(columns=["source", "target", "distance"])
    print("Skipped weighted shortest distance calculation.")


    def _process_flat_sbm(state, sbm_type_str, graph, all_block_summaries, all_node_centralities, all_inter_block_metrics):
        blocks = state.get_blocks()
        num_blocks = blocks.a.max() + 1
        print(f"\nProcessing {num_blocks} blocks from {sbm_type_str} SBM...")
        for b_id in tqdm(range(num_blocks), desc=f"Analyzing {sbm_type_str} Blocks"):
            vfilt = blocks.a == b_id
            num_nodes_in_block = np.sum(vfilt)

            subgraph = gt.GraphView(graph, vfilt=vfilt)
            block_summary, node_cents = _calculate_block_metrics_internal(subgraph, graph.vp.name, b_id, sbm_type_str)
            all_block_summaries.append(block_summary)
            all_node_centralities.extend(node_cents)

        print(f"Finished processing {sbm_type_str} SBM blocks.")

        inter_block_metrics = calculate_inter_block_metrics(graph, state)
        inter_block_metrics['sbm_type'] = sbm_type_str
        all_inter_block_metrics = pd.concat([all_inter_block_metrics, inter_block_metrics], ignore_index=True)

        return all_inter_block_metrics


    flat_exp_state, _ = flat_exponential_result
    all_inter_block_metrics = _process_flat_sbm(flat_exp_state, "flat_exponential", graph, all_block_summaries, all_node_centralities, all_inter_block_metrics)

    flat_lognorm_state, _ = flat_lognormal_result
    all_inter_block_metrics = _process_flat_sbm(flat_lognorm_state, "flat_lognormal", graph, all_block_summaries, all_node_centralities, all_inter_block_metrics)

    flat_pp_state, _ = flat_pp_result
    all_inter_block_metrics = _process_flat_sbm(flat_pp_state, "flat_pp", graph, all_block_summaries, all_node_centralities, all_inter_block_metrics)


    if 'all_inter_block_metrics' not in locals() or not isinstance(all_inter_block_metrics, pd.DataFrame):
        all_inter_block_metrics = pd.DataFrame()

    nested_exp_state, _ = nested_exponential_result
    all_inter_block_metrics = _process_nested_sbm_level(
        graph, nested_exp_state, "nested_exponential",
        nested_block_summaries_by_level, nested_node_centralities_by_level,
        all_inter_block_metrics
    )

    nested_lognorm_state, _ = nested_lognormal_result
    all_inter_block_metrics = _process_nested_sbm_level(
        graph, nested_lognorm_state, "nested_lognormal",
        nested_block_summaries_by_level, nested_node_centralities_by_level,
        all_inter_block_metrics
    )

    summary_df = pd.DataFrame(all_block_summaries)
    centrality_df = pd.DataFrame(all_node_centralities)

    shortest_distance_df = pd.DataFrame(columns=["source", "target", "distance"])

    analysis_results = (summary_df, centrality_df, shortest_distance_df, all_inter_block_metrics, nested_block_summaries_by_level, nested_node_centralities_by_level)

    with open(cache_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    print(f"Saved analysis results to cache: {cache_file}")

    return analysis_results