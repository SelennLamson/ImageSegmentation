import numpy as np
from typing import *
import math
import random
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import image_segmentation as seg
import maxflow

class SuperPixel:
	def __init__(self, sid):
		self.sid: int = sid
		self.neighbours: Dict[Tuple[int, int], List[float, float]] = defaultdict(list)
		self.captured_neighbours: Dict[Tuple[int, int], List[float]] = defaultdict(list)

	def get_weighted_neigbours(self, divided_img, pixeliser):
		neighbours = []
		weights = []
		to_remove = []
		for i, (nei, ws) in enumerate(self.neighbours.items()):
			other_region = divided_img[nei]
			if other_region != 0:
				to_remove.append(nei)
				self.captured_neighbours[nei] = [w for _, w in ws]
			else:
				neighbours.append(nei)
				weights.append(sum(w for w, _ in ws) / 3)
				# weights.append(sum(ws) / len(ws))

		for to_rem in to_remove:
			del self.neighbours[to_rem]

		if len(neighbours) == 0:
			return None, None

		return neighbours, np.array(weights)

	def add_neighbours(self, x, y, divided_img, pixeliser):
		neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
		for nei in neighbours:
			if divided_img[nei] == 0:
				self.neighbours[nei].append(pixeliser.get_weights(x, y, *nei))

	def capture_pixel(self, coords, divided_img, pixeliser):
		divided_img[coords] = self.sid
		del self.neighbours[coords]
		self.add_neighbours(*coords, divided_img, pixeliser)


class SuperPixeliser:
	def __init__(self, source_img, nb_superpixels=10000, subdivide_size=100):
		self.nb_superpixels = nb_superpixels
		self.subdivide_size = subdivide_size

		self.source_img = source_img
		self.w = source_img.shape[0]
		self.h = source_img.shape[1]
		self.weights_vert_hard = None
		self.weights_hori_hard = None
		self.weights_vert = None
		self.weights_hori = None
		self.divided_img = np.zeros((self.w + 2, self.h + 2), dtype=int)
		self.divided_img[0, :] = -1
		self.divided_img[-1, :] = -1
		self.divided_img[:, 0] = -1
		self.divided_img[:, -1] = -1

		self.superpixels: List[SuperPixel] = []
		self.colors = []

	def initialize_weights(self, weights_vert, weights_hori, weights_vert_hard, weights_hori_hard):
		self.weights_vert = weights_vert
		self.weights_hori = weights_hori
		self.weights_vert_hard = weights_vert_hard
		self.weights_hori_hard = weights_hori_hard

	def initialize_seeds(self):
		density = self.nb_superpixels / (self.w * self.h)
		n_regions_w = int(math.ceil(self.w / self.subdivide_size))
		n_regions_h = int(math.ceil(self.h / self.subdivide_size))
		last_region_w = self.w % self.subdivide_size
		last_region_h = self.h % self.subdivide_size
		if last_region_w == 0:
			last_region_w = self.subdivide_size
		if last_region_h == 0:
			last_region_h = self.subdivide_size
		n_seeds = 0

		for region_x in range(n_regions_w):
			region_w = self.subdivide_size if region_x < n_regions_w - 1 else last_region_w
			region_start_x = region_x * self.subdivide_size + 1
			region_end_x = region_start_x + region_w

			for region_y in range(n_regions_h):
				region_h = self.subdivide_size if region_y < n_regions_h - 1 else last_region_h
				region_start_y = region_y * self.subdivide_size + 1
				region_end_y = region_start_y + region_h

				area = region_w * region_h
				region_seeds = int(math.ceil(area * density))

				for _ in range(region_seeds):
					seed_x = seed_y = None
					if seed_x is None or self.weights_vert[seed_x, seed_y] * self.weights_hori[seed_x, seed_y] < 0.5:
						seed_x = random.randint(region_start_x, region_end_x - 1)
						seed_y = random.randint(region_start_y, region_end_y - 1)

					seed_id = n_seeds + 1

					superpixel = SuperPixel(seed_id)
					self.colors.append(self.source_img[seed_x - 1, seed_y - 1])
					superpixel.add_neighbours(seed_x, seed_y, self.divided_img, self)
					self.superpixels.append(superpixel)

					self.divided_img[seed_x, seed_y] = seed_id

					n_seeds += 1

	def grow_superpixels(self, verbose=True):
		it = 0
		total = self.h * self.w
		while np.count_nonzero(self.divided_img) != (self.h + 2) * (self.w + 2):
			it += 1
			for sp in self.superpixels:
				current_neighbours, current_weights = sp.get_weighted_neigbours(self.divided_img, self)

				if current_neighbours is None:
					continue

				# indices = np.argsort(current_weights)
				# for ni in indices[-int(math.ceil(0.3 * len(indices))):]:
				# 	sp.capture_pixel(current_neighbours[ni], self.divided_img, self)

				# indices = np.argsort(current_weights)
				# for ni in indices[-100:]:
				# 	sp.capture_pixel(current_neighbours[ni], self.divided_img, self)

				sample = np.random.random(len(current_neighbours))
				for ni in range(len(current_neighbours)):
					if sample[ni] <= current_weights[ni]:
						sp.capture_pixel(current_neighbours[ni], self.divided_img, self)

			# if it % 5 == 0:
			# 	self.plot(pause=True)

			if verbose:
				current = np.count_nonzero(self.divided_img) - self.h * 2 - self.w * 2
				print("\rSuper-Pixelisation: {:.2f}%".format(current / total * 100), end="")
		print("")

	def create_graph(self, w_if, w_ib):
		g = seg.Graph()

		g.add_node(0)	# Source
		g.add_node(1)	# Target
		for sp in self.superpixels:
			g.add_node(sp.sid + 1)	# Superpixel

		for sp1 in self.superpixels:
			i1 = sp1.sid
			for nei, ws in sp1.captured_neighbours.items():
				i2 = self.divided_img[nei]
				if i1 >= i2:
					continue
				g.add_edge(i1 + 1, i2 + 1, sum(ws))

		merge_source = set()
		merge_target = set()
		for x in range(self.w):
			for y in range(self.h):
				i = self.divided_img[x + 1, y + 1]
				g.add_edge(i + 1, 0, w_if[x, y])
				g.add_edge(i + 1, 1, w_ib[x, y])
				if w_if[x, y] > 50:
					merge_source.add(i + 1)
					self.colors[i - 1] = np.array([0, 0, 255])
				elif w_ib[x, y] > 50:
					merge_target.add(i + 1)
					self.colors[i - 1] = np.array([255, 0, 0])

		for node in merge_source:
			g.contract_edge(0, node)

		for node in merge_target:
			g.contract_edge(1, node)

		print("Superpixels:", len(self.superpixels))
		print("Nodes:", len(g))
		print("Pixel-nodes:", len(g.available_nodes))
		print("Edges:", sum(len(edges) for n, edges in g.nodes.items()) / 2)

		return g

	def create_maxflow_graph(self, w_if, w_ib):
		edges: Dict[Tuple(int, int), float] = defaultdict(float)
		terminal_edges: Dict[int, Tuple(float, float)] = defaultdict(lambda: (0, 0))

		for sp1 in self.superpixels:
			i1 = sp1.sid
			for nei, ws in sp1.captured_neighbours.items():
				i2 = self.divided_img[nei]
				if i1 != i2:
					edges[i1, i2] += sum(ws)
					edges[i2, i1] += sum(ws)

		for x in range(self.w):
			for y in range(self.h):
				i = self.divided_img[x + 1, y + 1]
				terminal_edges[i] = (terminal_edges[i][0] + w_if[x, y], terminal_edges[i][1] + w_ib[x, y])

				if w_if[x, y] > 500:
					self.colors[i - 1] = np.array([0, 0, 255])
				elif w_ib[x, y] > 500:
					self.colors[i - 1] = np.array([255, 0, 0])

		g = maxflow.Graph[float](len(self.superpixels), len(edges))
		nodes = g.add_nodes(len(self.superpixels))

		for (n1, n2), w in edges.items():
			g.add_edge(nodes[n1 - 1], nodes[n2 - 1], w, w)

		for node, (fw, bw) in terminal_edges.items():
			g.add_tedge(nodes[node - 1], fw, bw)

		return g, nodes

	def get_labeled_image(self, labels):
		for src_lbl in labels[0]:
			if src_lbl >= 2:
				self.colors[src_lbl - 2] = np.array([0, 0, 255])

		for tar_lbl in labels[1]:
			if tar_lbl >= 2:
				self.colors[tar_lbl - 2] = np.array([255, 0, 0])

		new_img = np.zeros_like(self.source_img)
		for x in range(self.w):
			for y in range(self.h):
				index = self.divided_img[x + 1, y + 1] - 1
				if index > -1:
					new_img[x, y, :] = self.colors[self.divided_img[x + 1, y + 1] - 1]
		return new_img

	def get_labeled_image_maxflow(self, graph, nodes):
		for i in range(len(self.superpixels)):
			if graph.get_segment(nodes[i]):
				self.colors[i] = np.array([255, 0, 0])
			else:
				self.colors[i] = np.array([0, 0, 255])

		new_img = np.zeros_like(self.source_img)
		for x in range(self.w):
			for y in range(self.h):
				index = self.divided_img[x + 1, y + 1] - 1
				if index > -1:
					new_img[x, y, :] = self.colors[self.divided_img[x + 1, y + 1] - 1]
		return new_img

	def plot(self, pause=False):
		new_img = np.zeros_like(self.source_img)
		for x in range(self.w):
			for y in range(self.h):
				index = self.divided_img[x + 1, y + 1] - 1
				if index > -1:
					new_img[x, y, :] = self.colors[self.divided_img[x + 1, y + 1] - 1]

		for sp in self.superpixels:
			for nei in sp.captured_neighbours.keys():
				new_img[nei[0] - 1, nei[1] - 1] = (0, 0, 0)


		# im_plot = np.mean(np.array([self.weights_hori[:, :-1], self.weights_vert[:-1, :]]), axis=0)[:, :, np.newaxis].repeat(3, axis=2) * 255
		# im_plot = (new_img[:-1, :-1, :] == 0) * im_plot + new_img[:-1, :-1]

		# plt.imshow(np.mean(np.array([self.weights_hori[:, :-1], self.weights_vert[:-1, :]]), axis=0).swapaxes(0, 1), cmap='gray', vmin=0, vmax=1)
		plt.imshow(self.source_img.swapaxes(0, 1))
		plt.imshow(new_img.swapaxes(0, 1).astype(np.uint8), alpha=0.8)
		if pause:
			plt.pause(0.01)
		else:
			plt.show()

	def get_weights(self, x1, y1, x2, y2):
		if x1 == x2:
			y = min(y1, y2)
			return self.weights_vert_hard[x1 - 1, y - 1], self.weights_vert[x1 - 1, y - 1]
		else:
			x = min(x1, x2)
			return self.weights_hori_hard[x - 1, y1 - 1], self.weights_hori[x - 1, y1 - 1]




