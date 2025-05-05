import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

import load_data
import build_graph
import plot_degree_distribution 

def validate_sbm_sampling():
    print("--- Validating SBM by Sampling ---")

    print("Loading data and building graph...")
    data_sets = load_data.main()
    country_to_country_data = next((ds for ds in data_sets if ds.get('name') == 'country_to_country'), None)

    if country_to_country_data is None:
        print("Error: 'country_to_country' dataset not found.")
        return

    name, graph = build_graph.build_graph(country_to_country_data)

    if graph is None or graph.num_vertices() == 0:
        print("Error: Graph building failed or resulted in an empty graph.")
        return

    print(f"Graph built: {graph.num_vertices()} vertices, {graph.num_edges()} edges.")

    print("Loading or running SBM analysis...")
    CACHE_DIR = "./cache"
    graph_id = f"{graph.num_vertices()}_{graph.num_edges()}"
    sbm_cache_file = os.path.join(CACHE_DIR, f"{name}_{graph_id}_sbm_results.pkl")

    sbm_results = None
    if os.path.exists(sbm_cache_file):
        try:
            with open(sbm_cache_file, 'rb') as f:
                sbm_results = pickle.load(f)
            print("Successfully loaded cached SBM results.")
        except Exception as e:
            print(f"⚠️ Error loading cached SBM results: {e}. You may need to re-run main.py to generate them.")

    if sbm_results is None:
         print("SBM results not available. Please run main.py first to generate and cache the SBM results.")
         return

    try:
        best_state, best_dl, best_type, flat_exponential_result, flat_lognormal_result, nested_exponential_result, nested_lognormal_result = sbm_results
        print(f"Loaded best-fitting SBM state: {best_type}")
    except (ValueError, TypeError) as e:
        print(f"Error unpacking SBM results: {e}")
        return

    if best_state is None:
        print("Error: Best-fitting SBM state is None. Cannot sample.")
        return

    print("Sampling networks from the fitted blockmodel...")
    num_samples = 5
    sampled_graphs = []

    try:
        for i in range(num_samples):
            print(f"  Sampling network {i+1}/{num_samples}...")
            sampled_graph = best_state.sample_graph()
            sampled_graphs.append(sampled_graph)
            print(f"  Sampled graph {i+1}: {sampled_graph.num_vertices()} vertices, {sampled_graph.num_edges()} edges.")
    except Exception as e:
        print(f"Error during network sampling: {e}")
        return

    def plot_sampled_graphs(graphs, num_to_plot=2):
        print(f"\nPlotting the first {num_to_plot} sampled graphs...")
        for i, graph_to_plot in enumerate(graphs[:num_to_plot]):
            try:
                output_filename = f"sampled_graph_{i+1}.png"
                pos = gt.sfdp_layout(graph_to_plot)
                gt.graph_draw(graph_to_plot, pos=pos, output=output_filename, output_size=(500, 500))
                print(f"Sampled graph {i+1} plot saved to {output_filename}")
            except Exception as e:
                print(f"Error plotting sampled graph {i+1}: {e}")

    # Plot a few sampled graphs
    plot_sampled_graphs(sampled_graphs)


    print("\nCalculating metrics for real and sampled networks...")
    metrics_to_calculate = {
        "num_vertices": lambda g: g.num_vertices(),
        "num_edges": lambda g: g.num_edges(),
        "avg_degree": lambda g: g.num_edges() / g.num_vertices() if g.num_vertices() > 0 else 0,
        "avg_in_degree": lambda g: g.num_edges() / g.num_vertices() if g.num_vertices() > 0 else 0,
        "avg_out_degree": lambda g: g.num_edges() / g.num_vertices() if g.num_vertices() > 0 else 0,
        "global_clustering": lambda g: gt.global_clustering(g)[0] if g.num_edges() > 0 else 0,
        "reciprocity": lambda g: gt.edge_reciprocity(g) if g.num_edges() > 0 else 0,
        "avg_local_clustering": lambda g: np.mean(gt.local_clustering(g).a[np.isfinite(gt.local_clustering(g).a)].flatten()) if g.num_edges() > 0 and g.num_vertices() > 0 and gt.local_clustering(g).a[np.isfinite(gt.local_clustering(g).a)].size > 0 else 0,
        "modularity": lambda g, state: gt.modularity(g, state.get_blocks()) if state is not None else np.nan,
        "num_blocks": lambda g, state: state.get_B() if state is not None else np.nan,
    }

    metric_results = []

    print("  Calculating metrics for the real graph...")
    real_graph_metrics = {"network_type": "Real"}
    for metric_name, metric_func in metrics_to_calculate.items():
        try:
            if metric_name == "modularity":
                original_graph_blocks = best_state.get_blocks()

                real_graph_blocks_prop = graph.new_vertex_property("int")

                if graph.num_vertices() == best_state.g.num_vertices():
                     for i_v in range(graph.num_vertices()):
                          real_graph_blocks_prop[graph.vertex(i_v)] = original_graph_blocks[i_v]
                else:
                     print(f"    Warning: Real graph vertex count mismatch during block copy. This should not happen if the same graph is loaded.")
                     min_vertices = min(graph.num_vertices(), best_state.g.num_vertices())
                     for i_v in range(min_vertices):
                          real_graph_blocks_prop[graph.vertex(i_v)] = original_graph_blocks[i_v]


                real_graph_weights = None
                if 'weight' in graph.ep:
                    real_graph_weights = graph.ep.weight
                else:
                    print(f"    Warning: Real graph does not have a 'weight' edge property. Modularity calculation might be incorrect or fail.")

                calculated_value = gt.modularity(graph, real_graph_blocks_prop, weight=real_graph_weights)
            elif metric_name == "num_blocks":
                calculated_value = best_state.get_B() if best_state is not None else np.nan
            else:
                if 'state' in metric_func.__code__.co_varnames:
                     calculated_value = metric_func(graph, best_state)
                else:
                     calculated_value = metric_func(graph)


            if isinstance(calculated_value, (int, float, np.number)):
                 real_graph_metrics[metric_name] = float(calculated_value)
            else:
                 real_graph_metrics[metric_name] = calculated_value
        except Exception as e:
            real_graph_metrics[metric_name] = f"Error: {e}"
            print(f"    Error calculating {metric_name} for real graph: {e}")
    metric_results.append(real_graph_metrics)

    # Calculate metrics for sampled graphs
    for i, sampled_graph in enumerate(sampled_graphs):
        print(f"  Calculating metrics for sampled graph {i+1}...")
        sampled_graph_metrics = {"network_type": f"Sampled {i+1}"}

        real_graph_blocks = best_state.get_blocks()

        sampled_graph_blocks = sampled_graph.new_vertex_property("int")
        if sampled_graph.num_vertices() == graph.num_vertices():
             for i_v in range(sampled_graph.num_vertices()):
                  sampled_graph_blocks[sampled_graph.vertex(i_v)] = real_graph_blocks[i_v]
        else:
             print(f"    Warning: Sampled graph {i+1} has a different number of vertices ({sampled_graph.num_vertices()}) than the real graph ({graph.num_vertices()}). Block assignments might be incorrect.")
             min_vertices = min(sampled_graph.num_vertices(), graph.num_vertices())
             for i_v in range(min_vertices):
                  sampled_graph_blocks[sampled_graph.vertex(i_v)] = real_graph_blocks[i_v]


        sampled_graph_weights = None
        if 'weight' in sampled_graph.ep:
             sampled_graph_weights = sampled_graph.ep.weight
        else:
             print(f"    Warning: Sampled graph {i+1} does not have a 'weight' edge property. Modularity calculation might be incorrect or fail.")


        for metric_name, metric_func in metrics_to_calculate.items():
             calculated_value = np.nan # Default value in case of error
             try:
                  if metric_name == "modularity":
                       if sampled_graph_blocks is not None:
                            calculated_value = gt.modularity(sampled_graph, sampled_graph_blocks, weight=sampled_graph_weights)
                       else:
                            calculated_value = np.nan
                  elif metric_name == "num_blocks":
                      calculated_value = best_state.get_B() if best_state is not None else np.nan
                  else:
                      if 'state' in metric_func.__code__.co_varnames:
                           calculated_value = metric_func(sampled_graph, best_state)
                      else:
                           calculated_value = metric_func(sampled_graph)


                  if isinstance(calculated_value, (int, float, np.number)):
                       sampled_graph_metrics[metric_name] = float(calculated_value)
                  else:
                       sampled_graph_metrics[metric_name] = calculated_value
             except Exception as e:
                  sampled_graph_metrics[metric_name] = f"Error: {e}"
                  print(f"    Error calculating {metric_name} for sampled graph {i+1}: {e}")
        metric_results.append(sampled_graph_metrics)

    metrics_df = pd.DataFrame(metric_results)
    print("\nMetric Comparison:")
    print(metrics_df)

    print("\nPlotting metric comparisons...")

    try:
        metrics_to_plot = ["num_vertices", "num_edges", "avg_degree", "avg_in_degree", "avg_out_degree", "global_clustering", "reciprocity", "avg_local_clustering", "modularity", "num_blocks"]
        fig, axes = plt.subplots(nrows=len(metrics_to_plot), figsize=(10, 4 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
             axes = [axes]

        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i]
            values = []
            labels = []
            for index, row in metrics_df.iterrows():
                 value = row[metric_name]
                 if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                      values.append(value)
                      labels.append(row['network_type'])
                 else:
                      print(f"  Skipping plotting for metric '{metric_name}' on network '{row['network_type']}' due to invalid value: {value}")


            print(f"  DEBUG Plotting: Metric='{metric_name}', Labels={labels}, Values={values}")
            if values:
                 ax.bar(labels, values, color=['blue'] + ['orange'] * num_samples)
                 ax.set_ylabel(metric_name.replace('_', ' ').title())
                 ax.set_title(f"Comparison of {metric_name.replace('_', ' ').title()}")
                 ax.grid(axis='y', linestyle='--', alpha=0.6)
            else:
                 ax.set_title(f"Comparison of {metric_name.replace('_', ' ').title()} (No valid data to plot)")
                 ax.text(0.5, 0.5, "No valid data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


        plt.tight_layout()
        plot_filename = "sbm_validation_metrics_comparison.png"
        plt.savefig(plot_filename)
        print(f"Metric comparison plot saved to {plot_filename}")
        plt.close()

    except Exception as e:
        print(f"Error during metric plotting: {e}")

    print("\nPlotting degree distributions...")
    try:
        print("  Plotting degree distributions for the real graph...")
        plot_degree_distribution.plot_degree_distributions(graph, "Real Graph")

        print("  Plotting degree distributions for sampled graphs...")
        for i, sampled_graph in enumerate(sampled_graphs):
            plot_degree_distribution.plot_degree_distributions(sampled_graph, f"Sampled Graph {i+1}")

        print("Degree distribution plots generated.")

    except Exception as e:
        print(f"Error during degree distribution plotting: {e}")


if __name__ == "__main__":
    validate_sbm_sampling()