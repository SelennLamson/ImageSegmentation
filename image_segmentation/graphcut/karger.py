import re
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time


class Karger:
	def __init__(self):
		self.nodes: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
		self.available_nodes = []
		self.groups: Dict[int, List[int]] = defaultdict(list)

	def __len__(self):
		return len(self.nodes)

	def __repr__(self):
		# Print the number of nodes
		result = "--- Graph: ---\n"
		result += str(self.nodes) + "\n"
		result += "-------------"
		return result

	def copy(self):
		# Copy the graph
		new_graph = Karger()
		new_graph.nodes = deepcopy(self.nodes)
		new_graph.groups = deepcopy(self.groups)
		new_graph.available_nodes = deepcopy(self.available_nodes)
		return new_graph

	def add_node(self, node):
		self.nodes[node] = defaultdict(float)
		self.groups[node] = [node]
		if node >= 2:
			self.available_nodes.append(node)

	def add_edge(self, n1, n2, w):
		self.nodes[n1][n2] += w
		self.nodes[n2][n1] += w

	def get_random_edge(self):
		n1 = random.choice(self.available_nodes)
		# n1, edges = list(self.nodes.items())[n1_id]

		# n1, edges = random.choice(self.nodes.keys())
		n2 = random.choice(list(self.nodes[n1].keys()))
		return n2, n1

		# return random.choice(list(self.edges.keys()))

		# keys = list(self.edges.keys())
		# weights = np.array([self.edges[k] for k in keys])
		# weights = weights / weights.sum()
		# return keys[np.random.choice(np.arange(len(keys)), 1, p=weights)[0]]

	def contract_edge(self, n1, n2):
		self.groups[n1] += self.groups[n2]

		l1 = self.nodes[n1]
		l1.pop(n2)
		self.available_nodes.remove(n2)
		removed_edges = self.nodes.pop(n2)

		for dst, w in removed_edges.items():
			if dst != n1:
				ldst = self.nodes[dst]
				ldst.pop(n2)
				ldst[n1] += w
				l1[dst] += w

	def perform_random_cut(self):
		h = self.copy()
		while len(h) > 2:
			h.contract_edge(*h.get_random_edge())
		return h

	def perform_karger(self, n_iter):
		avg_time = None
		best_cut = None
		best_labels = None
		for it in range(n_iter):
			now = time.time()

			h = self.perform_random_cut()
			cut = h.nodes[0][1]
			if best_cut is None or cut < best_cut:
				best_cut = cut
				best_labels = (h.groups[0], h.groups[1])

			elapsed = time.time() - now
			if avg_time is None:
				avg_time = elapsed
			else:
				avg_time = 0.9 * avg_time + 0.1 * elapsed
			print("\rKarger: {:.2f}%, best-cut: {}, time: {}ms".format(100 * it / n_iter, best_cut, avg_time * 1000), end="")
		print("")
		print('best labels: ', best_labels)
		return best_cut, best_labels

