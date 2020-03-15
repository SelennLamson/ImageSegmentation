
class Node:
	def __init__(self, index):
		self.index = index
		self.prev_edge_index = -1
		self.max_cap_to_here = 0
		self.dist = None

class Edge:
	def __init__(self, initial_vertex, terminal_vertex, inv_edge_index):
		self.initial_vertex = initial_vertex
		self.terminal_vertex = terminal_vertex
		self.capacity = None
		self.flow = None
		self.inv_edge_index: int = inv_edge_index

class BoykovKolmogorov:
	def __init__(self, width, height):
		self.eps = 1e-3
		self.w = width
		self.h = height
		self.nb_nodes = self.w * self.h + 2
		self.nodes: List[Node] = []
		self.edges: List[Edge] = []
		self.starting_edges: List[List[int]] = [None] * self.nb_nodes

		neighbours_to_create = [(1, 0), (0, 1)]

		# Creating nodes
		self.nodes.append(Node(0))	# Source
		self.nodes.append(Node(1))	# Target
		for x in range(self.w):
			for y in range(self.h):
				i = x * self.h + y + 2
				assert i == len(self.nodes)
				self.nodes.append(Node(i))	# Pixel


		# Creating starting edges array for source and target
		self.starting_edges[0] = [-1] * (self.nb_nodes - 2)
		self.starting_edges[1] = [-1] * (self.nb_nodes - 2)


		# Creating terminal edges
		for x in range(self.w):
			for y in range(self.h):
				i1 = x * self.h + y + 2

				# Node to source
				self.edges.append(Edge(i1, 0, len(self.edges)))
				self.starting_edges[i1].append(len(self.edges) - 1)

				# Source to node
				self.edges.append(Edge(0, i1, len(self.edges) - 2))
				self.starting_edges[0][i1 - 2] = len(self.edges) - 1

				# Node to target
				self.edges.append(Edge(i1, 1, len(self.edges)))
				self.starting_edges[i1].append(len(self.edges) - 1)

				# Target to node
				self.edges.append(Edge(1, i1, len(self.edges) - 2))
				self.starting_edges[1][i1 - 2] = len(self.edges) - 1

				# Neighbours
				for dx, dy in neighbours_to_create:
					nx = x + dx
					ny = y + dy
					if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h:
						continue
					i2 = nx * self.h + ny + 2

					# Node to neighbour
					self.edges.append(Edge(i1, i2, len(self.edges)))
					self.starting_edges[i1].append(len(self.edges) - 1)

					# Neighbour to node
					self.edges.append(Edge(i2, i1, len(self.edges) - 2))
					self.starting_edges[i2].append(len(self.edges) - 1)

		# Graph-cut algorithm variables
		self.orphans: List[int] = []
		self.active: List[int] = []
		self.is_in_s: List[bool] = []
		self.is_in_a: List[bool] = []

	def set_internal_weight(self, x1, y1, x2, y2, w):
		i1 = x1 * self.h + y1 + 2
		i2 = x2 * self.w + y2 + 2
		for edge_index in self.starting_edges[i1]:
			if self.edges[edge_index].terminal_vertex == i2:
				self.edges[edge_index].capacity = w
				self.edges[self.edges[edge_index].inv_edge_index].capacity = w

	def set_terminal_weights(self, w_source, w_target):
		for x in range(self.w):
			for y in range(self.h):
				i1 = x * self.h + y
				self.edges[self.starting_edges[0][i1]].capacity = w_source[x, y]
				self.edges[self.edges[self.starting_edges[0][i1]].inv_edge_index].capacity = w_source[x, y]
				self.edges[self.starting_edges[1][i1]].capacity = w_target[x, y]
				self.edges[self.edges[self.starting_edges[1][i1]].inv_edge_index].capacity = w_target[x, y]

	def reset_flow(self):
		for edge in self.edges:
			edge.flow = 0

	def do_cut(self):
		self.reset_flow()

		self.active = []
		self.active.append(0)

		self.orphans = []

		self.is_in_s = [False] * self.nb_nodes
		self.is_in_s[0] = True

		self.is_in_a = [False] * self.nb_nodes
		self.is_in_a[0] = True

		for node in self.nodes:
			node.prev_edge_index = -1

		while True:
			last_edge_index = self.growth_stage()
			if last_edge_index == -1:
				return
			self.augmentation_stage(last_edge_index)
			self.adoption_stage()

	def growth_stage(self) -> int:
		"""
		Phase 1: building a tree by setting the previous edge index on each node, finding path to target
		:return: last edge index of the path to the target, -1 if no path found
		"""
		if self.is_in_s[1]:
			return self.nodes[1].prev_edge_index

		while len(self.active) > 0:
			current_node_index = self.active[0]

			if not self.is_in_a[current_node_index]:
				del self.active[0]
				continue

			for edge_index in self.starting_edges[current_node_index]:
				current_edge = self.edges[edge_index]

				# Edge saturated
				if current_edge.capacity - current_edge.flow <= self.eps:
					continue

				# Edge leads to an non-active node, we expand the tree
				if not self.is_in_s[current_edge.terminal_vertex]:
					tv = current_edge.terminal_vertex
					self.active.append(tv)
					self.is_in_a[tv] = True
					self.is_in_s[tv] = True
					self.nodes[tv].prev_edge_index = edge_index

				# The non-active node was the target, we return the edge index to ther
				if current_edge.terminal_vertex == 1:
					return edge_index

			# Node treated, we remove it from active nodes and we go on
			del self.active[0]
			self.is_in_a[current_node_index] = False

		# No path found
		return -1

	def augmentation_stage(self, last_edge_index):
		"""
		Phase 2: saturating the path starting with the last edge index
		"""
		last_edge = self.edges[last_edge_index]
		bottle_neck_cap = last_edge.capacity - last_edge.flow

		current_node_index = last_edge.initial_vertex
		while current_node_index != 0:
			edge = self.edges[self.nodes[current_node_index].prev_edge_index]

			rem_flow = edge.capacity - edge.flow
			if bottle_neck_cap > rem_flow:
				bottle_neck_cap = rem_flow

			current_node_index = edge.initial_vertex

		while last_edge_index != -1:
			last_edge = self.edges[last_edge_index]

			# Adding flow to the edge
			last_edge.flow += bottle_neck_cap
			self.edges[last_edge.inv_edge_index].flow -= bottle_neck_cap

			# Next edge iteration
			last_edge_index = self.nodes[last_edge.initial_vertex].prev_edge_index

			# Orphan child if edge saturated
			if last_edge.capacity - last_edge.flow <= self.eps:
				self.nodes[last_edge.terminal_vertex].prev_edge_index = -1
				self.orphans.insert(0, last_edge.terminal_vertex)

	def adoption_stage(self):
		"""
		Phase 3: repairing the search tree by processing orphans
		"""
		while len(self.orphans) > 0:
			current_node_index = self.orphans.pop(0)
			found_parent = False

			# Searching for parent
			for edge_index in self.starting_edges[current_node_index]:
				current_edge = self.edges[self.edges[edge_index].inv_edge_index]

				# Edge is saturated
				if current_edge.capacity - current_edge.flow <= self.eps:
					continue

				# Node is not in source part of graph
				if not self.is_in_s[current_edge.initial_vertex]:
					continue

				current_root_node_index = current_edge.initial_vertex
				while self.nodes[current_root_node_index].prev_edge_index > 0:
					current_root_node_index = self.edges[self.nodes[current_root_node_index].prev_edge_index].initial_vertex

				# Node doesn't have the source as a root
				if current_root_node_index != 0:
					continue

				found_parent = True
				self.nodes[current_node_index].prev_edge_index = self.edges[edge_index].inv_edge_index
				break

			# A parent wasn't found for orphan
			if not found_parent:
				self.is_in_s[current_node_index] = False
				self.is_in_a[current_node_index] = False

				for edge_index in self.starting_edges[current_node_index]:
					current_edge = self.edges[edge_index]

					# Children become orphans
					tv = current_edge.terminal_vertex
					if self.nodes[tv].prev_edge_index == edge_index:
						self.nodes[tv].prev_edge_index = -1
						self.orphans.append(tv)

					if not self.is_in_s[tv]:
						continue

					current_edge = self.edges[current_edge.inv_edge_index]
					iv = current_edge.initial_vertex

					# Add in active nodes if not already there
					if current_edge.capacity - current_edge.flow > self.eps and not self.is_in_a[iv]:
						self.active.append(iv)
						self.is_in_a[iv] = True
