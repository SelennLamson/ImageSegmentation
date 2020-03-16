from typing import List, Tuple
import numpy as np

class BoykovKolmogorov:
	def __init__(self, width, height, source_weights, target_weights, hori_weights, vert_weights):
		self.eps = 1e-4
		self.w = width
		self.h = height
		self.nb_nodes = self.w * self.h + 2
		self.nb_edges = self.w * self.h * 4 + self.w * (self.h - 1) * 2 + (self.w - 1) * self.h * 2

		self.nodes_prev = np.ones((self.w, self.h, 2), dtype=int) * -1
		self.hori_edges = np.dstack([hori_weights, hori_weights])
		self.vert_edges = np.dstack([vert_weights, vert_weights])
		self.source_edges = source_weights.copy()
		self.target_edges = target_weights.copy()

		self.target_prev = (-1, -1)
		self.target_in_s = False

		self.orphans: List[Tuple[int, int]] = list()
		self.active: List[Tuple[int, int]] = list()

		self.is_in_s = np.zeros((self.w, self.h), dtype=bool)
		self.is_in_a = np.zeros((self.w, self.h), dtype=bool)

	def get_labeled_image(self) -> np.ndarray:
		print("Generating labelled image")

		blue = self.is_in_s * 255
		red = (1 - self.is_in_s) * 255
		img = np.dstack([blue, np.zeros((self.w, self.h)), red], dtype=np.uint8)
		return img

	def do_cut(self):
		self.active.append((-1, -1))

		it = 0
		while True:
			it += 1
			if it % 1000 == 0:
				print("\rIteration", it, "- In S:", np.sum(self.is_in_s), end="")

			last_x, last_y = self.growth_stage()
			if last_x == -1:
				return
			self.augmentation_stage(last_x, last_y)
			self.adoption_stage()

	def growth_stage(self) -> Tuple[int, int]:
		"""
		Phase 1: building a tree by setting the previous edge index on each node, finding path to target
		:return: last node (x, y) of the path to the target, (-1, -1) if no path found
		"""
		if self.target_in_s:
			return self.target_prev

		while len(self.active) > 0:
			current_node = self.active[0]

			if current_node == (-1, -1):
				for x in range(self.w):
					for y in range(self.h):
						if self.source_edges[x, y] > self.eps:
							if not self.is_in_s[x, y]:
								self.active.append((x, y))
								self.is_in_a[x, y] = True
								self.is_in_s[x, y] = True
								self.nodes_prev[x, y] = (-1, -1)
			else:
				x, y = current_node

				if not self.is_in_a[current_node]:
					del self.active[0]
					continue

				# Is connected to target?
				if self.target_edges[x, y] > self.eps:
					self.target_in_s = True
					self.target_prev = current_node
					return current_node

				# Connection to right neighbour
				if x < self.w - 1 and self.hori_edges[x, y, 0] > self.eps:
					nx, ny = x + 1, y
					if not self.is_in_s[nx, ny]:
						self.active.append((nx, ny))
						self.is_in_a[nx, ny] = True
						self.is_in_s[nx, ny] = True
						self.nodes_prev[nx, ny] = x, y

				# Connection to left neighbour
				if x > 0 and self.hori_edges[x - 1, y, 1] > self.eps:
					nx, ny = x - 1, y
					if not self.is_in_s[nx, ny]:
						self.active.append((nx, ny))
						self.is_in_a[nx, ny] = True
						self.is_in_s[nx, ny] = True
						self.nodes_prev[nx, ny] = x, y

				# Connection to bottom neighbour
				if y < self.h - 1 and self.vert_edges[x, y, 0] > self.eps:
					nx, ny = x, y + 1
					if not self.is_in_s[nx, ny]:
						self.active.append((nx, ny))
						self.is_in_a[nx, ny] = True
						self.is_in_s[nx, ny] = True
						self.nodes_prev[nx, ny] = x, y

				# Connection to top neighbour
				if y > 0 and self.vert_edges[x, y - 1, 1] > self.eps:
					nx, ny = x, y - 1
					if not self.is_in_s[nx, ny]:
						self.active.append((nx, ny))
						self.is_in_a[nx, ny] = True
						self.is_in_s[nx, ny] = True
						self.nodes_prev[nx, ny] = x, y

				# Node treated, we remove it from active nodes and we go on
				self.is_in_a[current_node] = False

			del self.active[0]

		# No path found
		return -1, -1

	def augmentation_stage(self, last_x, last_y):
		"""
		Phase 2: saturating the path starting with the last edge index
		"""
		bottle_neck_cap = self.target_edges[last_x, last_y]

		x, y = last_x, last_y
		reached_source = False
		while not reached_source:
			nx, ny = self.nodes_prev[x, y]

			rem_flow = bottle_neck_cap
			if nx == -1:
				rem_flow = self.source_edges[x, y]
				reached_source = True
			elif x == nx - 1:
				rem_flow = self.hori_edges[x, y, 0]
			elif x == nx + 1:
				rem_flow = self.hori_edges[nx, y, 1]
			elif y == ny - 1:
				rem_flow = self.vert_edges[x, y, 0]
			elif y == ny + 1:
				rem_flow = self.vert_edges[x, ny, 1]

			if bottle_neck_cap > rem_flow:
				bottle_neck_cap = rem_flow

			x, y = nx, ny

		x, y = last_x, last_y
		self.target_edges[x, y] -= bottle_neck_cap
		if self.target_edges[x, y] <= self.eps:
			self.orphans.append((-2, -2))
			self.target_prev = (-2, -2)

		reached_source = False
		while not reached_source:
			nx, ny = self.nodes_prev[x, y]

			if nx == -1:
				self.source_edges[x, y] -= bottle_neck_cap
				reached_source = True

				if self.source_edges[x, y] <= self.eps:
					self.nodes_prev[x, y] = -2, -2
					self.orphans.insert(0, (x, y))
			elif x == nx - 1:
				self.hori_edges[x, y, 0] -= bottle_neck_cap
				self.hori_edges[x, y, 1] += bottle_neck_cap

				if self.hori_edges[x, y, 0] <= self.eps:
					self.nodes_prev[x, y] = -2, -2
					self.orphans.insert(0, (x, y))
			elif x == nx + 1:
				self.hori_edges[nx, y, 1] -= bottle_neck_cap
				self.hori_edges[nx, y, 0] += bottle_neck_cap

				if self.hori_edges[nx, y, 1] <= self.eps:
					self.nodes_prev[x, y] = -2, -2
					self.orphans.insert(0, (x, y))
			elif y == ny - 1:
				self.vert_edges[x, y, 0] -= bottle_neck_cap
				self.vert_edges[x, y, 1] += bottle_neck_cap

				if self.vert_edges[x, y, 0] <= self.eps:
					self.nodes_prev[x, y] = -2, -2
					self.orphans.insert(0, (x, y))
			elif y == ny + 1:
				self.vert_edges[x, ny, 1] -= bottle_neck_cap
				self.vert_edges[x, ny, 0] += bottle_neck_cap

				if self.vert_edges[x, ny, 1] <= self.eps:
					self.nodes_prev[x, y] = -2, -2
					self.orphans.insert(0, (x, y))

			x, y = nx, ny

	def adoption_stage(self):
		"""
		Phase 3: repairing the search tree by processing orphans
		"""
		while len(self.orphans) > 0:
			x, y = self.orphans.pop(0)
			found_parent = False

			if x == -2:
				for nx in range(self.w):
					for ny in range(self.h):
						if self.target_edges[nx, ny] > self.eps:
							if self.is_in_s[nx, ny]:
								root_x, root_y = nx, ny
								while root_x >= 0:
									root_x, root_y = self.nodes_prev[root_x, root_y]
								if root_x == -1:
									found_parent = True
									self.target_prev = (nx, ny)
									break
					if found_parent:
						break

				if not found_parent:
					self.target_in_s = False

			else:
				# Is connected to source?
				if self.source_edges[x, y] > self.eps:
					self.nodes_prev[x, y] = -1, -1
					continue

				# Connection to right neighbour
				if x < self.w - 1 and self.hori_edges[x, y, 0] > self.eps:
					nx, ny = x + 1, y
					if self.is_in_s[nx, ny]:
						root_x, root_y = nx, ny
						while root_x >= 0:
							root_x, root_y = self.nodes_prev[root_x, root_y]
						if root_x == -1:
							self.nodes_prev[x, y] = nx, ny
							continue

				# Connection to left neighbour
				if x > 0 and self.hori_edges[x - 1, y, 1] > self.eps:
					nx, ny = x - 1, y
					if self.is_in_s[nx, ny]:
						root_x, root_y = nx, ny
						while root_x >= 0:
							root_x, root_y = self.nodes_prev[root_x, root_y]
						if root_x == -1:
							self.nodes_prev[x, y] = nx, ny
							continue

				# Connection to bottom neighbour
				if y < self.h - 1 and self.vert_edges[x, y, 0] > self.eps:
					nx, ny = x, y + 1
					if self.is_in_s[nx, ny]:
						root_x, root_y = nx, ny
						while root_x >= 0:
							root_x, root_y = self.nodes_prev[root_x, root_y]
						if root_x == -1:
							self.nodes_prev[x, y] = nx, ny
							continue

				# Connection to top neighbour
				if y > 0 and self.vert_edges[x, y - 1, 1] > self.eps:
					nx, ny = x, y - 1
					if self.is_in_s[nx, ny]:
						root_x, root_y = nx, ny
						while root_x >= 0:
							root_x, root_y = self.nodes_prev[root_x, root_y]
						if root_x == -1:
							self.nodes_prev[x, y] = nx, ny
							continue

				# No parent found
				self.is_in_s[x, y] = False
				self.is_in_a[x, y] = False

				# Orphan target if linked to this node
				if self.target_prev == (x, y):
					self.target_prev = -2, -2
					self.orphans.append((-2, -2))

				# Connection to right neighbour
				if x < self.w - 1 and self.hori_edges[x, y, 0] > self.eps:
					nx, ny = x + 1, y

					if self.nodes_prev[nx, ny, 0] == x and self.nodes_prev[nx, ny, 1] == y:
						self.nodes_prev[nx, ny] = -2, -2
						self.orphans.append((nx, ny))

					if self.is_in_s[nx, ny] and not self.is_in_a[nx, ny]:
						if self.hori_edges[x, y, 1] > self.eps:
							self.active.append((nx, ny))
							self.is_in_a[nx, ny] = True

				# Connection to left neighbour
				if x > 0 and self.hori_edges[x - 1, y, 1] > self.eps:
					nx, ny = x - 1, y

					if self.nodes_prev[nx, ny, 0] == x and self.nodes_prev[nx, ny, 1] == y:
						self.nodes_prev[nx, ny] = -2, -2
						self.orphans.append((nx, ny))

					if self.is_in_s[nx, ny] and not self.is_in_a[nx, ny]:
						if self.hori_edges[x - 1, y, 0] > self.eps:
							self.active.append((nx, ny))
							self.is_in_a[nx, ny] = True

				# Connection to bottom neighbour
				if y < self.h - 1 and self.vert_edges[x, y, 0] > self.eps:
					nx, ny = x, y + 1

					if self.nodes_prev[nx, ny, 0] == x and self.nodes_prev[nx, ny, 1] == y:
						self.nodes_prev[nx, ny] = -2, -2
						self.orphans.append((nx, ny))

					if self.is_in_s[nx, ny] and not self.is_in_a[nx, ny]:
						if self.vert_edges[x, y, 1] > self.eps:
							self.active.append((nx, ny))
							self.is_in_a[nx, ny] = True

				# Connection to top neighbour
				if y > 0 and self.vert_edges[x, y - 1, 1] > self.eps:
					nx, ny = x, y - 1

					if self.nodes_prev[nx, ny, 0] == x and self.nodes_prev[nx, ny, 1] == y:
						self.nodes_prev[nx, ny] = -2, -2
						self.orphans.append((nx, ny))

					if self.is_in_s[nx, ny] and not self.is_in_a[nx, ny]:
						if self.vert_edges[x, y - 1, 0] > self.eps:
							self.active.append((nx, ny))
							self.is_in_a[nx, ny] = True
