import graph_tool.all as gt
import pickle
import os
import numpy as np

def analyze_blocks(graph, block_state, model_type):
    print(f"\n--- Analyzing Blocks for {model_type} Model ---")

    weighted_modularity = None
    weighted_assortativity = None

    # Ensure the graph has weights
    if "weight" not in graph.ep:
        print("  Error: 'weight' edge property not found on the graph. Cannot calculate weighted metrics.")
        return weighted_modularity, weighted_assortativity

    if model_type.startswith("nested"):
        partition = block_state.b[0]
        print("  Calculating Weighted Modularity for the top level of the nested partition.")
    else:
        partition = block_state.b
        print("  Calculating Weighted Modularity for the flat partition.")

    if model_type == 'flat_pp':
        print("  Skipping weighted modularity calculation for unweighted flat_pp model.")
        weighted_modularity = None
    else:
        try:
            uv = gt.GraphView(graph, directed=False)
            weighted_modularity = gt.modularity(uv, partition, weight=uv.ep.weight)
            print(f"  Weighted Modularity: {weighted_modularity}")
        except Exception as e:
            print(f"  Error calculating Weighted Modularity: {e}")
            weighted_modularity = None


    print("  Skipping weighted assortativity calculation by block partition in analyze_blocks.")
    weighted_assortativity = None

    return weighted_modularity, weighted_assortativity



