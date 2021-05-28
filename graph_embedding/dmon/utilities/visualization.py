import operator
from typing import List

import matplotlib.pyplot as plt
import networkx as nx


def show_results_on_graph(graph: nx.Graph, predictions: List, frame_no: str):
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))

    ax.set_title('DMON Training Results on Salsa Cocktail Party')

    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
                       '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    node_pos = {node_n: feat[:2] for feat, node_n in
                zip(nx.get_node_attributes(graph, 'feats').values(),
                    nx.get_node_attributes(graph, 'person_no'))}

    node_edgecolors = ['black'] * graph.number_of_nodes()
    linewidths = [1 if c == 'black' else 5 for c in node_edgecolors]

    nx.draw(
        graph,
        node_color=list(nx.get_node_attributes(graph, 'color').values()),
        pos=node_pos,
        linewidths=linewidths,
        width=.3, ax=ax, node_size=200, edgecolors=node_edgecolors)

    predicted_node_colors = [distinct_colors[i] for i in predictions]
    for color, feature in zip(predicted_node_colors, nx.get_node_attributes(graph, 'feats').values()):
        pos = feature[:2]
        new_pos = list(map(operator.add, pos, [0.05, 0.2]))
        plt.scatter(*new_pos, 200, color=color, edgecolors='black', linewidths=1, alpha=0.4)

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.savefig(f'dmon_{frame_no}.png')
    return
