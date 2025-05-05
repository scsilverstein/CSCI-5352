import pandas as pd
import numpy as np
import graph_tool.all as gt

def build_graph(data_set):
    name = data_set['name']
    data = data_set['data']
    source_column = 'COUNTRY' if name == 'group_to_country' else 'COUNTERPART_COUNTRY'
    target_column = 'COUNTERPART_COUNTRY' if name == 'group_to_country' else 'COUNTRY'
    data = data.rename(columns={"2023": "weight"})

    data['weight'] = pd.to_numeric(data['weight'])

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
    return name,graph
