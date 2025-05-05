import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import load_data
import build_graph
import pandas as pd

def plot_us_import_distributions():
    print("--- Plotting US Import Weight Distributions ---")

    print("Loading data...")
    data_sets = load_data.main()
    country_to_country_data = next((ds for ds in data_sets if ds.get('name') == 'country_to_country'), None)

    if country_to_country_data is None:
        print("Error: 'country_to_country' dataset not found.")
        return

    print("Building graph...")
    name, graph = build_graph.build_graph(country_to_country_data)

    if graph is None or graph.num_vertices() == 0:
        print("Error: Graph building failed or resulted in an empty graph.")
        return

    if "name" not in graph.vp:
        print("Error: 'name' vertex property not found in graph.")
        return

    us_vertex = None
    for v in graph.vertices():
        country_name = graph.vp.name[v]
        if country_name == "United States" or country_name == "USA":
            us_vertex = v
            break

    if us_vertex is None:
        print("Error: 'United States' or 'USA' vertex not found in the graph.")
        print("Please verify the exact name used for the United States in the dataset.")
        return

    print(f"Found US vertex: {graph.vp.name[us_vertex]} (index: {int(us_vertex)})")

    import_data = []
    for e in us_vertex.in_edges():
        source_vertex = e.source()
        source_country = graph.vp.name[source_vertex]
        if "weight" in graph.ep:
            weight = graph.ep.weight[e]
            if np.isfinite(weight):
                 import_data.append({'country': source_country, 'weight': float(weight)})
            else:
                 print(f"Warning: Skipping edge from {source_country} due to non-finite weight: {weight}")
        else:
            print("Warning: 'weight' edge property not found. Cannot extract import weights.")
            return

    if not import_data:
        print("No import data found for the United States.")
        return

    import_df = pd.DataFrame(import_data)
    import_df_sorted = import_df.sort_values(by='weight', ascending=False).reset_index(drop=True)

    print(f"Found {len(import_df_sorted)} import sources for the United States.")

    top_n = 20
    import_df_top_n = import_df_sorted.head(top_n)

    if import_df_top_n.empty:
        print(f"No import data to plot for the top {top_n} sources.")
        return

    plt.figure(figsize=(12, 8))
    plt.bar(import_df_top_n['country'], import_df_top_n['weight'])
    plt.xticks(rotation=90)
    plt.xlabel("Source Country")
    plt.ylabel("Import Weight")
    plt.title(f"Top {top_n} US Import Sources by Weight")
    plt.tight_layout()

    output_filename = "us_import_weight_distribution.png"
    plt.savefig(output_filename)
    print(f"US import distribution plot saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    plot_us_import_distributions()