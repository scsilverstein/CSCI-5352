import graph_tool.all as gt
import numpy as np
import os
import pickle


def _prepare_lognormal_graph(graph):
    graph_copy = graph.copy()
    log_weights = graph_copy.new_edge_property("double")
    original_weights = graph_copy.ep.weight
    for e in graph_copy.edges():
        w = original_weights[e]
        log_weights[e] = np.log(w)

    graph_copy.ep.weight = log_weights

    valid_edge_filter = graph_copy.new_edge_property("bool")
    for e in graph_copy.edges():
        is_finite = np.isfinite(graph_copy.ep.weight[e])
        valid_edge_filter[e] = is_finite

    graph_copy.set_edge_filter(valid_edge_filter)

    return graph_copy


def _format_metric(metric):
    return f"{metric:.2f}"


def _run_sbm_model(graph, model_type, dist_type, cache_file, use_weights=True):
    if model_type == "flat_pp":
        model_name = "Flat PP (Unweighted)"
        dist_type = "unweighted"
        use_weights = False
    elif use_weights and dist_type:
         model_name = f"{model_type.capitalize()} {dist_type.capitalize()}"
    elif not use_weights:
         model_name = f"{model_type.capitalize()} Unweighted"
         dist_type = "unweighted"
    else:
         model_name = f"{model_type.capitalize()} UnknownDist"
         dist_type = "unknown"

    print(f"--- Running/Loading {model_name} SBM ---")

    base, ext = os.path.splitext(cache_file)
    if dist_type not in base:
        base = base.replace(model_type, f"{model_type}_{dist_type}")
    cache_file = base + ext

    if os.path.exists(cache_file):
        print(f"  Cache found for {model_name}. Loading results...")
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        print(f"  Loaded cached {model_name} results.")
        return result

    graph_to_use = graph
    state_args = dict(deg_corr=True)

    if use_weights:
        rec_type = f"real-{dist_type}"
        state_args['recs'] = [graph.ep.weight]
        state_args['rec_types'] = [rec_type]

        if dist_type == "lognormal":
            prepared_graph = _prepare_lognormal_graph(graph)
            graph_to_use = prepared_graph
            state_args['recs'] = [graph_to_use.ep.weight]
            state_args['rec_types'] = ["real-normal"]
        elif model_type == "flat" and dist_type == "exponential":
             # Apply log1p and filter zero weights for flat exponential
             graph_copy = graph.copy()
             log1p_weights = graph_copy.new_edge_property("double")
             original_weights = graph_copy.ep.weight
             for e in graph_copy.edges():
                  w = original_weights[e]
                  log1p_weights[e] = np.log1p(w)

             graph_copy.ep.weight = log1p_weights

             valid_edge_filter = graph_copy.new_edge_property("bool")
             for e in graph_copy.edges():
                  is_positive = graph_copy.ep.weight[e] > 0
                  valid_edge_filter[e] = is_positive

             graph_to_use = gt.GraphView(graph_copy, efilt=valid_edge_filter)
             state_args['recs'] = [graph_to_use.ep.weight] 
             state_args['rec_types'] = ["real-exponential"] 


    minimize_args = {}
    if model_type == "flat":
        minimize_func = gt.minimize_blockmodel_dl
    elif model_type == "nested":
        minimize_func = gt.minimize_nested_blockmodel_dl
    elif model_type == "flat_pp":
        minimize_func = gt.minimize_blockmodel_dl
        minimize_args['state'] = gt.PPBlockState
    else:
        print(f"  Error: Unknown model_type '{model_type}'")
        return None, np.inf

    print(f"  Running {model_name} minimization...")
    effective_state_args = state_args if len(state_args) > 1 or use_weights else {}

    call_args = dict(state_args=effective_state_args, multilevel_mcmc_args=dict(niter=100))
    call_args.update(minimize_args)

    result_state = minimize_func(graph_to_use, **call_args)
    result_metric = result_state.entropy()
    print(f"  {model_name} minimization complete. DL/Entropy: {result_metric}")

    final_result = (result_state, result_metric)
    with open(cache_file, 'wb') as f:
        pickle.dump(final_result, f)
    print(f"  Saved {model_name} results to cache: {cache_file}")

    return final_result



def compare_wsbm_models(flat_exponential_result, flat_lognormal_result, flat_pp_result, nested_exponential_result, nested_lognormal_result):
    def _safe_unpack(result):
        return result[0], result[1]

    flat_exp_state, flat_exp_dl = _safe_unpack(flat_exponential_result)
    flat_lognorm_state, flat_lognorm_dl = _safe_unpack(flat_lognormal_result)
    flat_pp_state, flat_pp_dl = _safe_unpack(flat_pp_result)
    nested_exp_state, nested_exp_dl = _safe_unpack(nested_exponential_result)
    nested_lognorm_state, nested_lognorm_dl = _safe_unpack(nested_lognormal_result)


    print("\n--- Model Comparison ---")
    print(f"  Flat Exponential Model DL: {_format_metric(flat_exp_dl)}")
    print(f"  Flat Log-Normal Model DL: {_format_metric(flat_lognorm_dl)}")
    print(f"  Flat PP (Unweighted) Model DL: {_format_metric(flat_pp_dl)}")
    print(f"  Nested Exponential Model DL: {_format_metric(nested_exp_dl)}")
    print(f"  Nested Log-Normal Model DL: {_format_metric(nested_lognorm_dl)}")


    best_dl = np.inf
    best_state = None
    best_type = "none"

    best_dl = flat_exp_dl
    best_state = flat_exp_state
    best_type = "flat_exponential"

    if flat_lognorm_dl < best_dl:
        best_dl = flat_lognorm_dl
        best_state = flat_lognorm_state
        best_type = "flat_lognormal"

    if flat_pp_dl < best_dl:
        best_dl = flat_pp_dl
        best_state = flat_pp_state
        best_type = "flat_pp"

    if nested_exp_dl < best_dl:
        best_dl = nested_exp_dl
        best_state = nested_exp_state
        best_type = "nested_exponential"

    if nested_lognorm_dl < best_dl:
        best_dl = nested_lognorm_dl
        best_state = nested_lognorm_state
        best_type = "nested_lognormal"

    print(f"  Best model is: {best_type} with DL: {best_dl}")
    return best_state, best_dl, best_type


def minimize_graph(graph):
    print("\n--- Running Bayesian WSBM with Model Selection ---")

    cache_dir = ".sbm_cache"
    os.makedirs(cache_dir, exist_ok=True)

    graph_id = f"{graph.num_vertices()}_{graph.num_edges()}"

    flat_exponential_result = _run_sbm_model(
        graph, "flat", "exponential",
        os.path.join(cache_dir, f"{graph_id}_flat_exp.pkl")
    )
    flat_lognormal_result = _run_sbm_model(
        graph, "flat", "lognormal",
        os.path.join(cache_dir, f"{graph_id}_flat_lognorm.pkl")
    )
    flat_pp_result = _run_sbm_model(
        graph, "flat_pp", "unweighted",
        os.path.join(cache_dir, f"{graph_id}_flat_pp_unweighted.pkl"),
        use_weights=False
    )
    nested_exponential_result = _run_sbm_model(
        graph, "nested", "exponential",
        os.path.join(cache_dir, f"{graph_id}_nested_exp.pkl")
    )
    nested_lognormal_result = _run_sbm_model(
        graph, "nested", "lognormal",
        os.path.join(cache_dir, f"{graph_id}_nested_lognorm.pkl")
    )


    best_state, best_dl, best_type = compare_wsbm_models(
        flat_exponential_result,
        flat_lognormal_result,
        flat_pp_result,
        nested_exponential_result,
        nested_lognormal_result
    )

    print(f"\n--- Best Model: {best_type} ---")
    return best_state, best_dl, best_type, flat_exponential_result, flat_lognormal_result, flat_pp_result, nested_exponential_result, nested_lognormal_result