import re
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time


class Graph:
	def __init__(self):
		self.nodes: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
		self.available_nodes = []
		self.groups: Dict[int, List[int]] = defaultdict(list)

	def __len__(self):
		return len(self.nodes)

	def nb_edges(self):
		return sum(len(edges) for node, edges in self.nodes.items()) // 2

	def __repr__(self):
		result = "--- Graph: ---\n"
		result += str(self.nodes) + "\n"
		result += "-------------"
		return result

	def copy(self):
		new_graph = Graph()
		new_graph.nodes = deepcopy(self.nodes)
		return new_graph

	def add_node(self, node):
		self.nodes[node] = defaultdict(float)
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
		# self.groups[n1].append(n2)

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

	# def load_graph(self, file_name):
	# 	file = open(file_name, 'r')
	# 	lines = file.readlines()
	# 	file.close()
	#
	# 	lines = map(lambda s: re.sub(r'\s+', ' ', str(s.strip('\r\n'))).strip(), lines)
	# 	lines = list(map(lambda s: s.split(' '), lines))
	#
	# 	for line in lines:
	# 		self.add_node(int(line[0]))
	#
	# 	for line in lines:
	# 		src = int(line[0])
	# 		for dst in list(map(lambda s: int(s), line[1:])):
	# 			self.add_edge(src, dst, 1)

	# def random_graph(self, n_nodes, edge_prob):
	# 	for ni in range(n_nodes):
	# 		self.add_node(ni)
	#
	# 	for u in range(n_nodes):
	# 		for v in range(u + 1, n_nodes):
	# 			if random.random() <= edge_prob:
	# 				self.add_edge(u, v, 1)

	def random_grid_graph(self, side):
		# Create graph, as we would like
		G = nx.grid_2d_graph(side, side)

		# Assign weights to graph above
		weights = list(np.random.random(G.number_of_edges()))  # random weight on edges

		# Return set of edges and nodes
		list_of_nodes = list(G.nodes())

		self.add_node(0)
		self.add_node(1)

		for x, y in list_of_nodes:
			nid = 2 + x + y * side
			self.add_node(nid)
			self.add_edge(0, nid, 1)
			self.add_edge(1, nid, 1)

		for ei, e in enumerate(G.edges()):
			n1 = 2 + e[0][0] + e[0][1] * side
			n2 = 2 + e[1][0] + e[1][1] * side
			self.add_edge(n1, n2, weights[i])




n = 10
lengths = np.zeros(n)

avg_time = None

for i in range(n):
	g = Graph()
	# g.load_graph('kargerMinCut.txt')
	g.random_grid_graph(200)
	while len(g) > 2:
		re = g.get_random_edge()

		now = time.time()
		g.contract_edge(*re)
		elapsed = time.time() - now

		if elapsed is not None:
			if avg_time is None:
				avg_time = elapsed
			else:
				avg_time = 0.01 * elapsed + 0.99 * avg_time

		print("\r", len(g), '{:.3f}us'.format(avg_time * 1000000), end="")

	print("\n")
	lengths[i] = min([1 if not isinstance(node, Node) else len(node.subnodes) for node in g.nodes])

plt.hist(lengths)
plt.show()
