# Import libraries
import networkx as nx
import random
from typing import Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time


class AlexKarger:
    def __init__(self):
        pass

    def contract_edge(self, g: nx.Graph, edge: Tuple[Any, Any]):
        """
        @g: the graph
        @edge: the edge that needs to be contracted
        Returns: new graph with contracted edge,
        taking weight into consideration
        Remark: we do not sample an edge wrt to weight size (not conform to original algo)
        """
        # Create a copy the graph (expensive)
        h = g.copy()

        # Identify the nodes relative to the edge
        u, v = edge

        # Keep track of the previous contractions
        contraction = []
        contraction += g.nodes(data=True)[u]['contraction']
        contraction += g.nodes(data=True)[v]['contraction']

        # Compute new edge for the edges that requires it
        new_edges = defaultdict(lambda: 0)
        for x, y, d in g.edges([u, v], data=True):
            if y != u and y != v:
                new_edges[(u, y)] += d['weight']

        # Remove this two nodes
        h.remove_nodes_from([u, v])

        # Create new node that keeps track of the contractions
        h.add_node(u, contraction=contraction)

        # Add weights
        h.add_weighted_edges_from([(k[0], k[1], w) for k, w in new_edges.items()])

        return h

    def karger_mincut(self, g: nx.Graph, n_iter: int):
        """
        @g: graph
        @n_iter: number of times the algo is applied
        Returns: cut applied and its value.
        """
        best_graph = None
        best_cut = None

        # Repeat several times
        for _ in tqdm(range(n_iter)):
            h = g.copy()
            start_time = time.time()
            while h.number_of_nodes() > 2:
                print("\r", h.number_of_nodes(), h.number_of_edges(), end="")
                # Choose an edge uniformly at randomly
                e = random.choice(list(h.edges))
                # Add constraint on target and source node
                un = set(h.nodes(data=True)[e[0]]['contraction']).union(h.nodes(data=True)[e[1]]['contraction'])
                if set(['target', 'source']).issubset(set(un)):
                    pass
                else:
                    h = contract_edge(h, e)
            end_time = time.time()
            total_time = end_time - start_time
            print("Time: ", total_time)

            # Store the best cut among iterations
            cut = h.get_edge_data(*list(h.edges)[0])['weight']
            if best_cut is None or cut < best_cut:
                best_cut = cut
                best_graph = h
        print("Min-cut :", best_cut)

        best_labels = []
        for node in best_graph.nodes(data=True):
            best_labels.append(node[1]['contraction'])

        return best_cut, best_labels
