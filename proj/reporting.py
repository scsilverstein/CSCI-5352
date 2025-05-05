import pandas as pd
import numpy as np
import graph_tool.all as gt


def _report_blocks(report_lines, state_name, state, original_graph):
    report_lines.append(f"\n--- Block Assignments for {state_name} ---" )
    print(f"\n--- Block Assignments for {state_name} ---")
    vertex_names = original_graph.vertex_properties["name"]
    if isinstance(state, gt.NestedBlockState):
        report_lines.append(f"  Nested Block Model:"  )

        print("  Nested Block Model:")
        for level_idx, level in enumerate(state.levels):
            report_lines.append(f"    Level {level_idx}:" )

            print(f"    Level {level_idx}:")
            blocks = level.get_blocks()
            block_members = {}
            for v in original_graph.vertices():
                block = blocks[v]
                if block not in block_members:
                    block_members[block] = []
                block_members[block].append(vertex_names[v])

            for block_id, members in sorted(block_members.items()):
                report_lines.append(f"      Block {block_id}: {', '.join(sorted(members))}" )

                print(f"      Block {block_id}: {', '.join(sorted(members))}")
    else: 
        report_lines.append(f"  Flat Block Model: " + state_name)

        print("  Flat Block Model:")
        blocks = state.get_blocks()
        block_members = {}
        for v in original_graph.vertices(): 
            block = blocks[v]
            if block not in block_members:
                block_members[block] = []
            block_members[block].append(vertex_names[v])

        for block_id, members in sorted(block_members.items()):
            report_lines.append(f"    Block {block_id}: {', '.join(sorted(members))}")

            print(f"    Block {block_id}: {', '.join(sorted(members))}")
    return report_lines

def _format_metric(value, precision=4):
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return f"{value:.{precision}f}"
    elif isinstance(value, str):
         return value 
    else:
        return "N/A" 

def _report_model_fit(report_lines, model_name, result):
    if result is not None and len(result) > 1 and result[1] is not None and np.isfinite(result[1]):
        report_lines.append(f"  {model_name} DL: {_format_metric(result[1], 4)}")
    else:
        report_lines.append(f"  {model_name}: Not fitted or failed.")

def _report_metric_stats(report_lines, df, metric, label, precision=4):
    if metric in df.columns and not df[metric].isnull().all():
        stats = df[metric].agg(['min', 'max', 'mean', 'std']).apply(lambda x: _format_metric(x, precision))
        report_lines.append(f"  {label}: Min={stats['min']}, Max={stats['max']}, Avg={stats['mean']}, Std Dev={stats['std']}")
    elif metric in df.columns:
        report_lines.append(f"  {label}: All NaN or no data.")

def _report_top_n_blocks(report_lines, df, metric, label, n=None, ascending=False):
    if metric not in df.columns or df[metric].isnull().all():
        report_lines.append(f"  No valid {metric} data for notable blocks.")
        return
    
    top_blocks = df.sort_values(by=metric, ascending=ascending).head(n)
    if not top_blocks.empty:
        report_lines.append(f"  {label}:")
        for _, row in top_blocks.iterrows():
            block_id = row.get('block_id', 'N/A')
            size = row.get('num_vertices', 'N/A')
            value = _format_metric(row[metric], 4)
            report_lines.append(f"    - Block {block_id} (Size: {size}): {metric.replace('_', ' ').title()} {value}")
    else:
         report_lines.append(f"  No blocks found for {label}.") 



def generate_summary_report(name, graph, best_state, summary_df, centrality_df, shortest_distance_df, inter_block_metrics_df, global_metrics=None, weighted_modularity=None, weighted_assortativity=None, nested_block_summaries_by_level=None, nested_node_centralities_by_level=None, flat_exponential_result=None, flat_lognormal_result=None, flat_pp_result=None, nested_exponential_result=None, nested_lognormal_result=None, best_type=None, best_dl=None):
    report_lines = []
    report_lines.append(f"--- Summary Report for Dataset: {name} ---")
    report_lines.append("=" * (30 + len(name)))

    report_lines.append("\n[SBM Structure and Model Comparison]")

    # Use helper to report model fits
    report_lines.append("\nSBM Model Fit (Description Length / Log-Likelihood):")
    results = {"Flat Exponential":flat_exponential_result,
               "Flat Log-Normal":flat_lognormal_result,
               "Nested Exponential": nested_exponential_result,
               "Nested Log-Normal":nested_lognormal_result}
    for label, result in results.items():
        _report_model_fit(report_lines, label, result)
    
    report_lines.append("\n Weighted Modularity")
    for label, result in results.items():
        if isinstance(result[0], gt.NestedBlockState):
            for level in result[0].get_levels():
                mod = gt.modularity(level.g, level.get_blocks())
                report_lines.append(label+"level: " +_format_metric(mod,4))
                print(level.g.ep)
                

        else:
            mod = gt.modularity(result[0].g, result[0].get_blocks(), weight=result[0].g.ep.weight)
            report_lines.append(label+": " +_format_metric(mod,4))
    
    
    report_lines.append("\n Weighted Assortivity")
    for label, result in results.items():
        if isinstance(result[0], gt.NestedBlockState):
            for level in result[0].get_levels():
                assort = gt.assortativity(level.g, level.get_blocks())
                report_lines.append(label+"level: " +_format_metric(assort,4))
                

        else:
            assort = gt.assortativity(result[0].g, result[0].get_blocks(), eweight=result[0].g.ep.weight)
            report_lines.append(label+": " +_format_metric(assort,4))
       
    report_lines.append("\n Weighted Local Cluster Coef")
    for label, result in results.items():
        if isinstance(result[0], gt.NestedBlockState):
            for level in result[0].get_levels():
                clocal = gt.local_clustering(level.g)

                report_lines.append(label+"level: " +_format_metric(clocal,4))
                

        else:
            clocal = gt.local_clustering(result[0].g, weight=result[0].g.ep.weight)
            report_lines.append(label+": " +_format_metric(clocal,4))
    
    if flat_exponential_result and flat_exponential_result[0]:
        _report_blocks(report_lines, "Flat Exponential", flat_exponential_result[0], graph)
    if flat_lognormal_result and flat_lognormal_result[0]:
        _report_blocks(report_lines, "Flat Log-Normal", flat_lognormal_result[0], graph)
    if flat_pp_result and flat_pp_result[0]:
        _report_blocks(report_lines, "Flat PP (Unweighted)", flat_pp_result[0], graph)
    if nested_exponential_result and nested_exponential_result[0]:
        _report_blocks(report_lines, "Nested Exponential", nested_exponential_result[0], graph)
    if nested_lognormal_result and nested_lognormal_result[0]:
        _report_blocks(report_lines, "Nested Log-Normal", nested_lognormal_result[0], graph)

    if best_type and best_dl is not None and np.isfinite(best_dl):
        report_lines.append(f"\nBest Fitting Model: {best_type} (DL: {_format_metric(best_dl, 4)})")
    else:
        report_lines.append("\nCould not determine a best fitting model.")


    # Report number of blocks for each flat model
    if flat_exponential_result is not None and flat_exponential_result[0] is not None:
        total_flat_exp_blocks = flat_exponential_result[0].get_blocks().a.max() + 1
        assigned_exp_blocks = set(flat_exponential_result[0].get_blocks().a)
        non_empty_exp_blocks = len(assigned_exp_blocks)
        report_lines.append(f"\nFlat Exponential SBM Blocks Found: {total_flat_exp_blocks} (Non-Empty: {non_empty_exp_blocks})")
    else:
        report_lines.append("\nFlat Exponential SBM: Not fitted or failed.")

    if flat_lognormal_result is not None and flat_lognormal_result[0] is not None:
        total_flat_lognorm_blocks = flat_lognormal_result[0].get_blocks().a.max() + 1
        assigned_lognorm_blocks = set(flat_lognormal_result[0].get_blocks().a)
        non_empty_lognorm_blocks = len(assigned_lognorm_blocks)
        report_lines.append(f"Flat Log-Normal SBM Blocks Found: {total_flat_lognorm_blocks} (Non-Empty: {non_empty_lognorm_blocks})")
    else:
        report_lines.append("Flat Log-Normal SBM: Not fitted or failed.")

    if flat_pp_result is not None and flat_pp_result[0] is not None:
        total_flat_pp_blocks = flat_pp_result[0].get_blocks().a.max() + 1
        assigned_pp_blocks = set(flat_pp_result[0].get_blocks().a)
        non_empty_pp_blocks = len(assigned_pp_blocks)
        report_lines.append(f"Flat PP (Unweighted) SBM Blocks Found: {total_flat_pp_blocks} (Non-Empty: {non_empty_pp_blocks})")
    else:
        report_lines.append("Flat PP (Unweighted) SBM: Not fitted or failed.")


    report_lines.append("\nNested SBM Block Counts by Level and Model:")
    nested_models_found = False
    for model_name, result in results.items():
        if result and result[0] and isinstance(result[0], gt.NestedBlockState):
            nested_models_found = True
            state = result[0]
            report_lines.append(f"  {model_name}:")
            if not state.levels:
                 report_lines.append("    - No levels found in this nested model.")
                 continue
            for level_idx, level in enumerate(state.get_levels()):
                try:
                    total_blocks = level.get_blocks().a.max() + 1
                    assigned_blocks = set(level.get_blocks().a)
                    non_empty_blocks = len(assigned_blocks)
                    report_lines.append(f"    - Level {level_idx}: {total_blocks} blocks (Non-Empty: {non_empty_blocks})")
                except ValueError:
                    report_lines.append(f"    - Level {level_idx}: Error calculating block counts.")

    if nested_models_found:
        report_lines.append("\nRefer to the hierarchical visualizations (circular plot and dendrogram) for a visual representation of this structure.")
    else:
        report_lines.append("  No nested SBM models were successfully fitted or provided.")




    report_lines.append("\n[Global Network Metrics]")
    if global_metrics:
        report_lines.append(f"  Density: {_format_metric(global_metrics.get('density'), 6)}")
        report_lines.append(f"  Global Clustering Coefficient: {_format_metric(global_metrics.get('global_clustering_coefficient'), 6)}")

        report_lines.append(f"  Global  Unweighted Clustering Coefficient: {_format_metric(global_metrics.get('global_unweighted_clustering_coefficient'), 6)}")
        report_lines.append(f"  Reciprocity: {_format_metric(global_metrics.get('reciprocity'), 6)}")
        report_lines.append(f"  Weighted Reciprocity: {_format_metric(global_metrics.get('weighted_reciprocity'), 6)}")
        report_lines.append(f"  Degree Assortativity: {_format_metric(global_metrics.get('degree_assortativity'), 6)}")
        report_lines.append(f"  Weighted Assortativity: {_format_metric(global_metrics.get('weighted_assortativity'), 6)}")
        report_lines.append(f"  Avg Weighted Shortest Path: {_format_metric(global_metrics.get('avg_weighted_shortest_path'), 6)}")
    else:
        report_lines.append("  Global metrics were not calculated or provided.")

    report_lines.append("\n[Weighted Block Metrics (for Best Model)]")
    if best_type and (weighted_modularity is not None or weighted_assortativity is not None):
        report_lines.append(f"  Metrics for Best Model ({best_type.replace('_', ' ').title()}):")
        report_lines.append(f"    Weighted Modularity: {_format_metric(weighted_modularity, 6)}")
        report_lines.append(f"    Weighted Assortativity (by block): {_format_metric(weighted_assortativity, 6)}")
        report_lines.append(f"    Weighted Assortativity (by block): {_format_metric(weighted_assortativity, 6)}")
    elif best_type:
        report_lines.append(f"  Weighted block metrics not available for the best model ({best_type.replace('_', ' ').title()}).")
    else:
        report_lines.append("  Best model not determined, cannot report weighted block metrics.")


    report_lines.append("\n[Block Statistics by Model]")
    if not summary_df.empty:
        sbm_types = summary_df['sbm_type'].unique()

        for sbm_type in sorted(sbm_types):
            model_summary = summary_df[summary_df['sbm_type'] == sbm_type]
            num_blocks = len(model_summary)

            if num_blocks > 0:
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}:")
                report_lines.append(f"  Blocks Found: {num_blocks}")
                non_empty_blocks = model_summary[model_summary['num_vertices'] > 0]
                if not non_empty_blocks.empty:
                    _report_metric_stats(report_lines, non_empty_blocks, 'num_vertices', "Block Sizes (Non-Empty)", precision=2)
                    
                    _report_metric_stats(report_lines, non_empty_blocks, 'density', "Density (Non-Empty Blocks)")
                    _report_metric_stats(report_lines, non_empty_blocks, 'avg_local_clustering', "Avg Local Clustering (Non-Empty Blocks)")
                    _report_metric_stats(report_lines, non_empty_blocks, 'reciprocity', "Reciprocity (Non-Empty Blocks)")
                    _report_metric_stats(report_lines, non_empty_blocks, 'assortativity', "Assortativity (Non-Empty Blocks)")
                else:
                    report_lines.append("  No non-empty blocks to report statistics for.")
            else:
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}: No blocks found.")
    else:
        report_lines.append("  No block summary data available for any model.")


    report_lines.append("\n[Notable Blocks (Examples - Top 3 by Density and Avg Clustering per Model)]")
    if not summary_df.empty:
        sbm_types = summary_df['sbm_type'].unique()
        for sbm_type in sorted(sbm_types):
            model_summary = summary_df[summary_df['sbm_type'] == sbm_type]
            if not model_summary.empty:
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}:")
                
                model_summary_filtered = model_summary[model_summary['num_vertices'] > 0]

                if not model_summary_filtered.empty:
                    _report_top_n_blocks(report_lines, model_summary_filtered, 'density', "Highest Density Blocks", n=None, ascending=False)
                    _report_top_n_blocks(report_lines, model_summary_filtered, 'avg_local_clustering', "Highest Avg Clustering Blocks", n=None, ascending=False)
                else:
                    report_lines.append("  No non-empty blocks to report notable blocks for.")
            else:
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}: No block summary data available.")
    else:
        report_lines.append("  No block summary data available for any model.")


    report_lines.append("\n[Top Central Nodes within Blocks (PageRank - Top 3 per Block per Model)]")
    if not centrality_df.empty:
        sbm_types = centrality_df['sbm_type'].unique()
        for sbm_type in sorted(sbm_types):
            model_centrality_df = centrality_df[centrality_df['sbm_type'] == sbm_type]
            if not model_centrality_df.empty and 'pagerank' in model_centrality_df.columns and not model_centrality_df['pagerank'].isnull().all():
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()} Blocks:")
                block_ids = model_centrality_df['block_id'].unique()
                for b_id in sorted(block_ids):
                    top_nodes = model_centrality_df[(model_centrality_df['block_id'] == b_id) & (model_centrality_df['pagerank'].notnull())].sort_values(by='pagerank', ascending=False).head(None)
                    if not top_nodes.empty:
                        report_lines.append(f"  Block {b_id}:")
                        for _, node_row in top_nodes.iterrows():
                            report_lines.append(f"    - {node_row['vertex_name']} (PR: {node_row['pagerank']:.4f})")
            elif not model_centrality_df.empty:
                 report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}: No valid PageRank data available.")
            else:
                 report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}: No centrality data available.")
    else:
        report_lines.append("  No centrality data available for any model.")


    report_lines.append("\n[Potential Transit Hubs (High Betweenness vs. PageRank - Based on Best Model)]")
    if best_type and not centrality_df.empty and 'betweenness' in centrality_df.columns and 'pagerank' in centrality_df.columns:
        best_model_centrality_df = centrality_df[centrality_df['sbm_type'] == best_type].dropna(subset=['betweenness', 'pagerank']).copy()

        if not best_model_centrality_df.empty:
            best_model_centrality_df['betweenness_rank'] = best_model_centrality_df['betweenness'].rank(method='min', ascending=False)
            best_model_centrality_df['pagerank_rank'] = best_model_centrality_df['pagerank'].rank(method='min', ascending=False)

            best_model_centrality_df['rank_diff'] = best_model_centrality_df['pagerank_rank'] - best_model_centrality_df['betweenness_rank']

            potential_hubs = best_model_centrality_df.sort_values(by='rank_diff', ascending=False).head(None)

            if not potential_hubs.empty:
                report_lines.append(f"  Top candidates in {best_type.replace('_', ' ').title()} (higher Betweenness rank than PageRank rank):")
                for _, hub_row in potential_hubs.iterrows():
                    report_lines.append(f"    - {hub_row['vertex_name']} (Block: {hub_row['block_id']}, Btw Rank: {int(hub_row['betweenness_rank'])}, PR Rank: {int(hub_row['pagerank_rank'])}, Diff: {hub_row['rank_diff']:.0f})")
            else:
                report_lines.append(f"  No significant rank differences found in {best_type.replace('_', ' ').title()} based on current criteria.")
        else:
             report_lines.append(f"  Not enough valid centrality data in {best_type.replace('_', ' ').title()} to perform rank comparison.")
    elif best_type:
        report_lines.append(f"  Betweenness or PageRank data missing for {best_type.replace('_', ' ').title()}, cannot perform transit hub analysis.")
    else:
        report_lines.append("  Best model not determined, cannot perform transit hub analysis.")


    report_lines.append("\n[Inter-Block Metrics Summary by Model]")
    if not inter_block_metrics_df.empty:
        sbm_types = inter_block_metrics_df['sbm_type'].unique()
        for sbm_type in sorted(sbm_types):
            model_inter_metrics = inter_block_metrics_df[inter_block_metrics_df['sbm_type'] == sbm_type]
            if not model_inter_metrics.empty:
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()} Inter-Block Trade:")
                
                def _report_trade_pairs(report_lines, df, metric, label, n=None, ascending=False):
                     if metric not in df.columns or df[metric].isnull().all():
                         report_lines.append(f"  No valid {metric} data for trade pairs.")
                         return
                     
                     filtered_df = df[df[metric] > 0] if ascending else df
                     
                     top_pairs = filtered_df.sort_values(by=metric, ascending=ascending).head()
                     if not top_pairs.empty:
                         report_lines.append(f"  {label}:")
                         for _, row in top_pairs.iterrows():
                             source = row.get('source_block', 'N/A')
                             target = row.get('target_block', 'N/A')
                             weight = _format_metric(row.get('total_weight'), 2)
                             edges = row.get('num_edges', 'N/A')
                             report_lines.append(f"    - Block {source} -> Block {target}: Total Weight {weight} ({edges} edges)")
                     elif ascending and not df[df[metric] > 0].empty:
                          report_lines.append(f"  No non-zero trade pairs found for {label}.")
                     elif ascending:
                          report_lines.append(f"  No trade data available for {label}.")
                     else:
                          report_lines.append(f"  No trade pairs found for {label}.")

                _report_trade_pairs(report_lines, model_inter_metrics, 'total_weight', "Top Block Pairs by Total Export Value", n=None, ascending=False)
                _report_trade_pairs(report_lines, model_inter_metrics, 'total_weight', "Bottom Block Pairs by Total Export Value (non-zero)", n=None, ascending=True)

            else:
                report_lines.append(f"\n{sbm_type.replace('_', ' ').title()}: No inter-block metrics data available.")
    else:
        report_lines.append("  No inter-block metrics data available for any model.")


    report_lines.append("\n" + "=" * (30 + len(name)))
    report_lines.append("--- End of Report ---")

    return "\n".join(report_lines)