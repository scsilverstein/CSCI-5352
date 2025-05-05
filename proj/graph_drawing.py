import graph_tool.all as gt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def draw_flat_sbm(state, graph, name):
    print(f"Drawing flat SBM graph for {name}...")
    output_filename = f"{name}_flat_blocks_mdl.png"
    state.draw(
        output=output_filename,
        vertex_text=state.g.vp.name,
        vertex_text_position="centered",
        vertex_font_size=8,
        vertex_text_offset=[0, 0],
        output_size=(1200, 1200),
        fit_view=True,
    )
    print(f"Flat SBM graph saved to {output_filename}")

def draw_nested_sbm(state, graph, name):
    print(f"Drawing nested SBM graph for {name}...")
    state.draw(output=f"{name}_nested_blocks_mdl.png", vertex_text=state.g.vp.name,
               vertex_text_position="centered", vertex_font_size=8,
               vertex_text_offset=[0, 0], output_size=(1200, 1200), fit_view=True)
    print(f"Successfully drew nested SBM graph for {name}.")

def draw_nested_sbm_levels(nested_blockmodel, graph, name):
    print(f"Drawing nested SBM levels for {name}...")
    level_labels = {}

    levels = nested_blockmodel.get_levels()

    for i, level in enumerate(levels):
        print(f"  Drawing Level {i}...")
        level_filename = f"{name}_nested_blocks_mdl_{i}.png"

        level_blocks = level.get_blocks()

        original_graph_blocks_at_level = nested_blockmodel.project_level(i).get_blocks()

        num_blocks_at_level = level.g.num_vertices()
        block_counts = np.bincount(original_graph_blocks_at_level.a, minlength=num_blocks_at_level)

        non_empty_block_indices = np.where(block_counts > 0)[0]

        vfilt_level = level.g.new_vertex_property("bool", val=False)
        for block_idx in non_empty_block_indices:
             vfilt_level[block_idx] = True

        filtered_level_graph = gt.GraphView(level.g, vfilt=vfilt_level)
        print(f"  Drawing filtered Level {i} graph ({filtered_level_graph.num_vertices()} non-empty blocks).")

        vp_name_level = filtered_level_graph.new_vertex_property("string")
        v_text = filtered_level_graph.new_vertex_property("string")
        filtered_block_membership = filtered_level_graph.new_vertex_property("int")

        for v_filtered in filtered_level_graph.vertices():
            v_level_orig = filtered_level_graph.vertex_index[v_filtered]
            filtered_block_membership[v_filtered] = level_blocks[v_level_orig]

            if i == 0:
                original_name = graph.vp.name[v_level_orig]
                vp_name_level[v_filtered] = original_name
                v_text[v_filtered] = original_name
            else:
                block_id = int(v_level_orig)

                prev_labels = level_labels.get(i-1)
                if prev_labels is None:
                     vp_name_level[v_filtered] = f"Block_{block_id}_Level_{i}"
                     v_text[v_filtered] = f"Block {block_id}"
                else:
                    prev_level_state = levels[i-1]
                    prev_block_mapping = prev_level_state.get_blocks()

                    members_labels = []
                    for v_prev in prev_level_state.g.vertices():
                         if prev_block_mapping[v_prev] == block_id:
                              members_labels.append(str(prev_labels[v_prev]))

                    vp_name_level[v_filtered] = f"Block_{block_id}_Level_{i}"
                    if members_labels:
                         if len(members_labels) > 3:
                             v_text[v_filtered] = f"Block {block_id}\n({', '.join(members_labels[:3])}...\n+{len(members_labels)-3} more)"
                         else:
                             v_text[v_filtered] = f"Block {block_id}\n({', '.join(members_labels)})"
                    else:
                         v_text[v_filtered] = f"Block {block_id} (No members found)"

        level_labels[i] = v_text

        pos = gt.sfdp_layout(filtered_level_graph, groups=filtered_block_membership,
                            C=5.0, K=15.0, p=2.0, theta=0.6,
                            max_iter=2000, gamma=1.5, mu=0.8)

        gt.graph_draw(
            filtered_level_graph,
            output=level_filename,
            vertex_shape=filtered_block_membership,
            vertex_text=v_text,
            vertex_text_position="centered",
            vertex_font_size=8,
            vertex_text_offset=[0, 0],
            pos=pos,
            output_size=(1200, 1200),
            fit_view=True,
            vertex_properties={"name": vp_name_level}
        )
        print(f"  Filtered Level {i} graph saved to {level_filename}")

def plot_block_metrics(summary_df, name, flat_exponential_result=None, flat_lognormal_result=None, nested_exponential_result=None, nested_lognormal_result=None):
    print(f"Generating block metric plots for {name} (all models)...")

    sbm_types = summary_df['sbm_type'].unique()

    for sbm_type in sbm_types:
        print(f"  Plotting metrics for {sbm_type}...")
        model_summary = summary_df[summary_df['sbm_type'] == sbm_type].copy()

        model_summary_filtered = model_summary[model_summary['num_vertices'] > 0]

        plt.figure(figsize=(10, 6))
        plt.scatter(model_summary_filtered['num_vertices'], model_summary_filtered['density'], alpha=0.7)
        plt.title(f'{name} - {sbm_type}: Block Density vs. Size (Non-Empty Blocks)')
        plt.xlabel('Number of Vertices in Block')
        plt.ylabel('Block Density')
        plt.grid(True, linestyle='--', alpha=0.6)
        plot_filename = f"{name}_{sbm_type}_block_density_vs_size.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Density vs. Size plot saved to {plot_filename}")

        plt.figure(figsize=(10, 6))
        num_bins = min(20, len(model_summary_filtered))
        plt.hist(model_summary_filtered['num_vertices'], bins=num_bins if num_bins > 0 else 1, edgecolor='black')
        plt.title(f'{name} - {sbm_type}: Distribution of Block Sizes (Non-Empty Blocks)')
        plt.xlabel('Number of Vertices in Block')
        plt.ylabel('Frequency (Number of Blocks)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plot_filename = f"{name}_{sbm_type}_block_size_hist.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Block Size histogram saved to {plot_filename}")

        valid_densities = model_summary_filtered['density'].dropna()
        plt.figure(figsize=(10, 6))
        plt.hist(valid_densities, bins=20, edgecolor='black')
        plt.title(f'{name} - {sbm_type}: Distribution of Block Densities (Non-Empty Blocks)')
        plt.xlabel('Block Density')
        plt.ylabel('Frequency (Number of Blocks)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plot_filename = f"{name}_{sbm_type}_block_density_hist.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Block Density histogram saved to {plot_filename}")

    print(f"Finished generating block metric plots for {name} (all models).")

def draw_block_matrix_heatmap(state, graph, name, matrix_type='adjacency', rec_type=None):
    print(f"Drawing block matrix heatmap for {name}...")

    if matrix_type == 'adjacency':
        full_matrix = state.get_matrix()
    elif matrix_type == 'membership':
        full_matrix = state.get_membership_matrix()
    else:
        print(f"  Warning: Unsupported matrix type '{matrix_type}'. Defaulting to adjacency.")
        full_matrix = state.get_matrix() # Default to adjacency if type is unsupported

    blocks = state.get_blocks()
    block_counts = np.bincount(blocks.a)

    non_empty_block_indices = np.where(block_counts > 0)[0]

    filtered_matrix = full_matrix[non_empty_block_indices, :][:, non_empty_block_indices]

    m = filtered_matrix.todense()
    print(f"  Drawing filtered block matrix heatmap ({len(non_empty_block_indices)}x{len(non_empty_block_indices)}).")
    min_value = np.min(m)
    max_value =np.max(m)
    print(f"Minimum value in the dense matrix: {min_value, max_value,m }")
    fig, ax = plt.subplots()
    mat = ax.matshow(m)

    plt.colorbar(mat, ax=ax, label='Value')

    num_blocks = m.shape[0]
    tick_positions = np.arange(num_blocks)

    block_labels = [str(idx) for idx in non_empty_block_indices]

    # Set ticks and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(block_labels, rotation=90, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(block_labels, fontsize=8)

    # Ensure ticks are centered on the cells for clarity (optional visual enhancement)
    # ax.set_xticks(np.arange(num_blocks+1)-.5, minor=True)
    # ax.set_yticks(np.arange(num_blocks+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=1) # Optional: Add grid lines
    # ax.tick_params(which="minor", bottom=False, left=False) # Hide minor ticks

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Block ID")
    ax.set_ylabel("Block ID")

    plt.title(f"Block Matrix Heatmap ({matrix_type})", pad=30)


    output_filename = f"{name}_{matrix_type}_matrix_heatmap.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"  Block matrix heatmap saved to {output_filename}")


def draw_hierarchical_visualization(nested_blockmodel, graph, name):
    print(f"Drawing hierarchical visualization for {name}...")

    output_filename_dendrogram = f"{name}_hierarchical_dendrogram.png"
    output_filename_circular = f"{name}_hierarchical_circular.png"

    levels = nested_blockmodel.get_levels()

    v_text = nested_blockmodel.g.vp.name

    gt.draw_hierarchy(nested_blockmodel, output=output_filename_circular,
                       vertex_text=v_text,
                       vertex_font_size=8,
                       vertex_size=10,
                       output_size=(1000, 1000),
                      )
    print(f"  Hierarchical circular bundled graph saved to {output_filename_circular}")


def plot_weight_distribution_fit(blockmodel, graph, name, rec_type):
    print(f"Plotting weight distribution fit for {name}...")
    output_filename = f"{name}_weight_distribution_fit.png"

    empirical_weights = graph.ep.weight.a[np.isfinite(graph.ep.weight.a)]

    model_predicted_weights = []
    blocks = blockmodel.get_blocks()
    num_blocks = blockmodel.get_B()
    state = blockmodel



    num_blocks = state.get_B()

    if rec_type == "real-exponential":
        rate_matrix = state.get_r()
        model_params = rate_matrix 
    elif rec_type == "real-normal":
        mean_matrix = state.get_r()
        variance_matrix = state.get_sigma()
        model_params = {"mu": mean_matrix, "sigma": variance_matrix} # Store in a dictionary
    else:
        print(f"  Warning: Unsupported rec_type '{rec_type}' for parameter access. Cannot sample weights.")
        return 

    print(f"Model parameters (accessed via state methods): {model_params}")

    num_samples = 10000

    for _ in range(num_samples):
        source_block = np.random.randint(0, num_blocks)
        target_block = np.random.randint(0, num_blocks)

        if rec_type == "real-exponential":
            rate = model_params[source_block, target_block]
            if rate > 0:
                 sampled_weight = np.random.exponential(scale=1.0/rate)
                 model_predicted_weights.append(sampled_weight)
        elif rec_type == "real-normal":
            mean_matrix = model_params["mu"]
            variance_matrix = model_params["sigma"]
            mean = mean_matrix[source_block, target_block]
            variance = variance_matrix[source_block, target_block]
            # Ensure variance is non-negative for sqrt
            if variance >= 0:
                 sampled_log_weight = np.random.normal(loc=mean, scale=np.sqrt(variance))
                 model_predicted_weights.append(sampled_log_weight)
            # else: skip sampling if variance is negative (shouldn't happen with valid fit)
        else:
             print(f"  Warning: Unsupported rec_type '{rec_type}' for weight sampling. Skipping.")
             continue

    plt.figure(figsize=(10, 6))

    plt.hist(empirical_weights, bins=50, density=True, alpha=0.6, label='Empirical')

    plt.hist(model_predicted_weights, bins=50, density=True, alpha=0.6, label='Model Predicted', color='orange')

    model_rec_type = "Unknown"
    if hasattr(blockmodel, 'state') and hasattr(blockmodel.state, 'rec_type'):
         model_rec_type = blockmodel.state.rec_type
    elif hasattr(blockmodel, 'rec_type'):
         model_rec_type = blockmodel.rec_type

    plt.title(f'{name} - Edge Weight Distribution Fit ({model_rec_type})')

    if model_rec_type == "real-normal":
         plt.xlabel('Log(Edge Weight)')
    else:
         plt.xlabel('Edge Weight')

    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"  Weight distribution fit plot saved to {output_filename}")
