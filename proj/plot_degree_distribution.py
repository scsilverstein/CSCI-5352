import graph_tool.all as gt

import matplotlib.pyplot as plt
import load_data
import build_graph
import numpy as np 

def plot_degree_distributions(graph, label):
    print(f"--- Plotting Degree Distributions for {label} ---")

    if graph.ep.get("weight") is not None:
        try:
            # counts, bins = gt.edge_hist(graph, graph.ep.weight)
            
            # print("Edge weights shape:", counts.shape)
            # print("Edge weight bins shape:", bins.shape)
            # # Print first few elements only if array is not empty
            # if counts.size > 0:
            #     print("Edge weights (first 10):", counts[:10])
            # if bins.size > 0:
            #     print("Edge weight bins (first 10):", bins[:10])

            # print("Edge weight distribution calculated.")
            weights = graph.ep.weight.a
            non_zero_weights = weights[weights > 0]
            plt.figure(figsize=(10, 6))
            # Plotting the first 500 bins as the range can be very large
            plt.hist(non_zero_weights, bins=1000, color='skyblue', alpha=0.7)
            plt.title(f"Edge Weight Distribution for {label}")
            plt.xlabel("Edge Weights (Bins)")
            
            plt.yscale('log') # Use log scale for frequency
            # plt.xscale('log') # Use log scale for weights
            # plt.xlim(left=1) # Set lower limit to 1 to avoid log(0)
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.6)
            filename = f"{label.replace(' ', '_').lower()}_edge_weight_distribution.png"
            plt.savefig(filename)
            print(f"Edge weight distribution plot saved to {filename}")
            # plt.show() # Commented out to prevent blocking execution
            plt.close()

            try:
                
                weights = graph.ep.weight.a
                non_zero_weights = weights[weights > 0]
                print("Non-zero edge weights shape:", non_zero_weights.shape)
                if non_zero_weights.size > 0:
                    
                    log_weights = np.log(non_zero_weights)

                    plt.figure(figsize=(10, 6))
                    plt.hist(log_weights, bins=100, color='lightcoral', alpha=0.7)
                    plt.title(f"Logarithm of Edge Weight Distribution for {label}", fontsize=16)
                    plt.xlabel("Log(Edge Weights)", fontsize=14)
                    plt.ylabel("Frequency", fontsize=14) # Changed ylabel to Frequency for consistency with unweighted degree
                    plt.grid(True, linestyle='--', alpha=0.6)
                    log_filename = f"{label.replace(' ', '_').lower()}_log_edge_weight_distribution.png"
                    plt.savefig(log_filename)
                    print(f"Logarithm of edge weight distribution plot saved to {log_filename}")
                    # plt.show() # Commented out to prevent blocking execution
                    plt.close()
                else:
                    print(f"    No non-zero edge weights to plot logarithm for {label}.")

            except Exception as e:
                print(f"Error plotting logarithm of edge weight distribution for {label}: {e}")


        except Exception as e:
            print(f"Error plotting edge weight distribution for {label}: {e}")


    for direction in ['in', 'out', 'total']:
        try:
            degree = graph.degree_property_map(direction)
            print(f"{direction.capitalize()} Degree property map shape:", degree.a.shape)
            if degree.a.size > 0:
                 print(f"{direction.capitalize()} Degree property map (first 10):", degree.a[:10])

            plt.figure(figsize=(10, 6))
            plt.hist(degree.a, bins=100, color='skyblue', alpha=0.7)
            plt.title(f"{direction.capitalize()} Degree Distribution for {label}", fontsize=16)
            plt.xlabel("Degree", fontsize=14)
            
            plt.ylabel("Frequency", fontsize=14) # Changed ylabel to Frequency for consistency with weighted degree
            plt.grid(True, linestyle='--', alpha=0.6)
            filename = f"{label.replace(' ', '_').lower()}_{direction}_degree_distribution.png"
            plt.savefig(filename)
            print(f"{direction.capitalize()} degree distribution plot saved to {filename}")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Error plotting {direction} degree distribution for {label}: {e}")

    if graph.ep.get("weight") is not None:
        for direction in ['in', 'out', 'total']:
            try:
                degree = graph.degree_property_map(direction, weight=graph.ep.weight)
                print(f"{direction.capitalize()} Weighted Degree property map shape:", degree.a.shape)
                if degree.a.size > 0:
                     print(f"{direction.capitalize()} Weighted Degree property map (first 10):", degree.a[:10])

                plt.figure(figsize=(10, 6))
                plt.hist(degree.a, bins=100, color='skyblue', alpha=0.7)
                plt.title(f"{direction.capitalize()} Weighted Degree Distribution for {label}", fontsize=16)
                plt.xlabel("Weighted Degree", fontsize=14)
                plt.ylabel("Frequency", fontsize=14) # Changed ylabel to Frequency for consistency with unweighted degree
                plt.grid(True, linestyle='--', alpha=0.6)
                filename = f"{label.replace(' ', '_').lower()}_{direction}_weighted_degree_distribution.png"
                plt.savefig(filename)
                print(f"{direction.capitalize()} weighted degree distribution plot saved to {filename}")
                plt.show()
                plt.close()
            except Exception as e:
                print(f"Error plotting {direction} weighted degree distribution for {label}: {e}")


if __name__ == "__main__":
    print("Running plot_degree_distributions standalone (requires data loading)...")
    data_sets = load_data.main()
    country_to_country_data = next((ds for ds in data_sets if ds.get('name') == 'country_to_country'), None)
    if country_to_country_data:
        name, graph = build_graph.build_graph(country_to_country_data)
        if graph and graph.num_vertices() > 0:
            plot_degree_distributions(graph, "Country-to-Country Graph")
        else:
            print("Could not build graph for standalone plotting.")
    else:
        print("'country_to_country' dataset not found for standalone plotting.")