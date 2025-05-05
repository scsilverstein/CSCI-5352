import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import load_data
import build_graph
import pandas as pd 

def plot_us_export_distributions():
    print("--- Plotting US Export Weight Distributions ---")

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

    export_data = []
    for e in us_vertex.out_edges():
        target_vertex = e.target()
        target_country = graph.vp.name[target_vertex]
        if "weight" in graph.ep:
            weight = graph.ep.weight[e]
            if np.isfinite(weight):
                 export_data.append({'country': target_country, 'weight': float(weight)})
            else:
                 print(f"Warning: Skipping edge to {target_country} due to non-finite weight: {weight}")
        else:
            print("Warning: 'weight' edge property not found. Cannot extract export weights.")
            return

    if not export_data:
        print("No export data found for the United States.")
        return

    export_df = pd.DataFrame(export_data)
    export_df_sorted = export_df.sort_values(by='weight', ascending=False).reset_index(drop=True)

    print(f"Found {len(export_df_sorted)} export destinations for the United States.")

    top_n = 20
    export_df_top_n = export_df_sorted.head(top_n)

    if export_df_top_n.empty:
        print(f"No export data to plot for the top {top_n} destinations.")
        return

    plt.figure(figsize=(12, 8))
    plt.bar(export_df_top_n['country'], export_df_top_n['weight'])
    plt.xticks(rotation=90)
    plt.xlabel("Target Country")
    plt.ylabel("Export Weight")
    plt.title(f"Top {top_n} US Export Destinations by Weight")
    plt.tight_layout()

    output_filename = "us_export_weight_distribution.png"
    plt.savefig(output_filename)
    print(f"US export distribution plot saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    plot_us_export_distributions()
