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
		self.source: List[float] = [0, 0]
		self.target: List[float] = [0, 0]

		self.available_nodes = []
		self.groups: Dict[int, List[int]] = defaultdict(list)

	def __len__(self):
		return len(self.nodes) + 2

	def __repr__(self):
		result = "--- Graph: ---\n"
		result += str(self.nodes) + "\n"
		result += "SRC: " + str(self.source) + "\n"
		result += "TAR: " + str(self.target) + "\n"
		result += "-------------"
		return result

	def add_node(self, node):
		self.nodes[node] = defaultdict(float)
		self.source.append(0)
		self.target.append(0)
		self.available_nodes.append(node)

	def add_edge(self, n1, n2, w):
		self.nodes[n1][n2] += w
		self.nodes[n2][n1] += w

	def add_source_edge(self, no, w):
		self.nodes[no][0] += w
		self.source[no] += w

	def add_target_edge(self, no, w):
		self.nodes[no][1] += w
		self.target[no] += w

	def get_random_edge(self):
		n1 = random.choice(self.available_nodes)
		n2 = random.choice(list(self.nodes[n1].keys()))
		return n1, n2

	def contract_edge(self, n1, n2):
		if n2 == 0:
			self.available_nodes.remove(n1)
			l1 = self.nodes.pop(n1)

			for dst, w in l1.items():
				if dst == 1:
					self.target[0] += w
					self.source[1] += w
				elif dst != 0:
					self.source[dst] += w

					ldst = self.nodes[dst]
					ldst.pop(n1)
					ldst[0] += w

		elif n2 == 1:
			self.available_nodes.remove(n1)
			l1 = self.nodes.pop(n1)

			for dst, w in l1.items():
				if dst == 0:
					self.target[0] += w
					self.source[1] += w
				elif dst != 1:
					self.target[dst] += w

					ldst = self.nodes[dst]
					ldst.pop(n1)
					ldst[1] += w

		else:
			# self.groups[n1].append(n2)

			l1 = self.nodes[n1]
			l1.pop(n2)
			self.available_nodes.remove(n2)
			removed_edges = self.nodes.pop(n2)

			for dst, w in removed_edges.items():
				if dst != n1:
					l1[dst] += w

					if dst == 0:
						self.source[n1] += w
					elif dst == 1:
						self.target[n1] += w
					else:
						ldst = self.nodes[dst]
						ldst.pop(n2)
						ldst[n1] += w

	def random_grid_graph(self, side):
		# Create graph, as we would like
		G = nx.grid_2d_graph(side, side)

		# Assign weights to graph above
		weights = list(np.random.random(G.number_of_edges()))  # random weight on edges

		# Return set of edges and nodes
		list_of_nodes = list(G.nodes())

		for ni in range(side * side):
			nid = ni + 2
			self.add_node(nid)
			self.add_source_edge(nid, 1)
			self.add_target_edge(nid, 1)

		for ei, e in enumerate(G.edges()):
			n1 = 2 + e[0][0] + e[0][1] * side
			n2 = 2 + e[1][0] + e[1][1] * side
			self.add_edge(n1, n2, weights[i])




n = 100
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
	lengths[i] = g.source[1]
	# lengths[i] = min([1 if not isinstance(node, Node) else len(node.subnodes) for node in g.nodes])

plt.hist(lengths)
plt.show()
