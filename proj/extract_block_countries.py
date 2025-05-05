
import pickle
import sys
import os
import pandas as pd
import numpy as np
import graph_tool.all as gt

def load_country_data(csv_filepath="dataset_exports_groupdescript.csv"):
    print("Loading data...")
    country_network_df = pd.read_csv(csv_filepath, low_memory=False)
    print('Data loaded')

    print("Extracting country to country data...")
    is_country_id = ~country_network_df["COUNTRY.ID"].astype(str).str.contains(r'\d', na=False)
    is_counterpart_country_id = ~country_network_df["COUNTERPART_COUNTRY.ID"].astype(str).str.contains(r'\d', na=False)
    country_to_country_network_df = country_network_df[is_country_id & is_counterpart_country_id].copy()
    print('Country to country data extracted')

    return {
        'name': 'country_to_country',
        'data': country_to_country_network_df
    }

def build_graph_from_data(data_set):
    name = data_set['name']
    data = data_set['data']

    source_column = 'COUNTRY' if name == 'group_to_country' else 'COUNTERPART_COUNTRY'
    target_column = 'COUNTERPART_COUNTRY' if name == 'group_to_country' else 'COUNTRY'
    data = data.rename(columns={"2023": "weight"})

    data['weight'] = pd.to_numeric(data['weight'], errors='coerce')

    source_countries = data[source_column].astype(str).unique()
    target_countries = data[target_column].astype(str).unique()
    all_countries = pd.unique(pd.concat([pd.Series(source_countries), pd.Series(target_countries)]))

    country_to_index = {name: i for i, name in enumerate(all_countries)}

    s = data[source_column].map(country_to_index).values
    t = data[target_column].map(country_to_index).values
    w = data['weight'].values

    es = np.array([s, t, w]).T

    graph = gt.Graph(directed=True)
    graph.add_vertex(len(all_countries))
    vertex_names = graph.new_vertex_property("string")
    for i, country in enumerate(all_countries):
        vertex_names[i] = country
    graph.vertex_properties["name"] = vertex_names

    graph.add_edge_list(es, eprops=[("weight", "double")])

    return graph

def extract_countries_from_sbm(sbm_pkl_filepath, output_txt_filepath):
    print(f"Loading SBM results from {sbm_pkl_filepath}...")
    with open(sbm_pkl_filepath, 'rb') as f:
        sbm_result = pickle.load(f)
    print("SBM results loaded.")

    state = sbm_result[0]

    print("Loading original graph to get country names...")
    country_data = load_country_data()
    graph = build_graph_from_data(country_data)
    print("Original graph loaded.")

    vertex_names = graph.vertex_properties["name"]

    levels = state.levels

    print(f"Writing block information to {output_txt_filepath}...")
    with open(output_txt_filepath, 'w') as f:
        for level_idx, level_state in enumerate(levels):
            f.write(f"--- Level {level_idx} ---\n")
            print(f"Processing Level {level_idx}...")

            blocks = level_state.get_blocks()

            unique_blocks = np.unique(blocks.a)

            for block_id in unique_blocks:
                f.write(f"  Block {block_id}:\n")
                print(f"  Processing Block {block_id}...")

                vertices_in_block = [v for v in graph.vertices() if blocks[v] == block_id]

                country_list = [vertex_names[v] for v in vertices_in_block]

                country_list.sort()

                for country in country_list:
                    f.write(f"    - {country}\n")
            f.write("\n")

    print(f"Successfully extracted and saved block countries to {output_txt_filepath}")


if __name__ == "__main__":
    sbm_pkl_filepath = sys.argv[1]
    output_txt_filepath = sys.argv[2]

    extract_countries_from_sbm(sbm_pkl_filepath, output_txt_filepath)