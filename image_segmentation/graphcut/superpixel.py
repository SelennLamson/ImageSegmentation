import numpy as np
from typing import *
import math
import random
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import image_segmentation as seg


class SuperPixel:
	def __init__(self, sid):
		# Superpixel id
		self.sid: int = sid
		# Free pixels on the frontier of this superpixel, and their associated weights (hard and soft weights)
		self.neighbours: Dict[Tuple[int, int], List[float, float]] = defaultdict(list)
		# Pixels on the frontier that already belongs to another superpixel, and their associated soft weights
		self.captured_neighbours: Dict[Tuple[int, int], List[float]] = defaultdict(list)

	def get_weighted_neigbours(self, divided_img, pixeliser):
		"""
		Computes a list of weighted pixels on the frontier of this superpixel, for growth.
		:return: neighbours and associated probability weights
		"""
		neighbours = []
		weights = []
		to_remove = []

		# Loop over all neighbours of a superpixel
		for i, (nei, ws) in enumerate(self.neighbours.items()):
			other_region = divided_img[nei]

			# Neighbour already belongs to another superpixel
			if other_region != 0:
				# We prepare to remove it after iterating over neighbours
				to_remove.append(nei)

				# We register it to the list of captured neighbours if it is not captured by this superpixel
				if other_region != self.sid:
					self.captured_neighbours[nei] += [w for _, w in ws]

			# Neighbour is free
			else:
				neighbours.append(nei)

				# The probability to capture a new pixel depends on the number of sides it is 'attacked' from, and of their respective weights
				weights.append(sum(w for w, _ in ws) / 4)

		# If pixel has just been captured by another superpixel, discard it from free neighbours
		for to_rem in to_remove:
			del self.neighbours[to_rem]

		# If no more free neighbours
		if len(neighbours) == 0:
			return None, None

		return neighbours, np.array(weights)

	def add_neighbours(self, x, y, divided_img, pixeliser):
		"""
		Tries to register the four neighbours of a given pixel as neighbours of the superpixel
		:param x: x-coordinate in the image
		:param y: y-coordinate in the image
		:param divided_img: array in which each cell contains the id of the superpixel, or 0 if not captured yet
		:param pixeliser: SuperPixeliser instance, to retrieve weights
		"""
		# List of all 4 neighbours of a point (x,y)
		neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
		for nei in neighbours:
			oid = divided_img[nei]

			# Select neighbour only if not already captured by another superpixel
			if oid == 0:
				self.neighbours[nei].append(pixeliser.get_weights(x, y, *nei))

			# If already captured by another superpixel, add it to list of captured neighbours
			elif oid != self.sid and oid != -1:
				self.captured_neighbours[nei].append(pixeliser.get_weights(x, y, *nei)[0])

	def capture_pixel(self, coords, divided_img, pixeliser):
		"""
		Captures a new pixel in this superpixel, adding its neighbours to the registers
		:param coords: (x,y) coordinates of the pixel
		:param divided_img: array in which each cell contains the id of the superpixel, or 0 if not captured yet
		:param pixeliser: SuperPixeliser instance, to retrieve weights
		"""
		# Attribute pixel to superpixel
		divided_img[coords] = self.sid

		# Delete pixel from neighbours, not available to capture anymore
		del self.neighbours[coords]

		# Try to register its neighbours
		self.add_neighbours(*coords, divided_img, pixeliser)


class SuperPixeliser:
	def __init__(self, source_img, nb_superpixels=10000, subdivide_size=100):
		"""
		Algorithm that splits an image in an approximate number of superpixels, respecting edges
		:param source_img: Source RGB/YUV image to split in superpixels
		:param nb_superpixels: Approximate number of superpixels to produce
		:param subdivide_size: Size of the regions in pixels to uniformly distribute the superpixels' seeds
		"""

		self.nb_superpixels = nb_superpixels  		# Objective number of superpixels
		self.subdivide_size = subdivide_size  		# Size of distribution regions
		self.source_img = source_img				# Source image to split
		self.w = source_img.shape[0]				# Image width
		self.h = source_img.shape[1]				# Image height
		self.superpixels: List[SuperPixel] = []		# List of superpixels
		self.colors = []							# Superpixels' color, for plotting purposed only

		# Init weights matrices
		self.weights_vert_hard = None				# Hard weights between pixels, vertically
		self.weights_hori_hard = None				# Hard weights between pixels, horizontally
		self.weights_vert = None					# Vertical weights (softer)
		self.weights_hori = None					# Horizontal weights (softer)

		# Array where each pixel's cell contains the ID of the superpixel it belongs to
		# An ID of 0 means that the pixel is not captured yet
		# An ID of -1 represents the border of the image and cannot be captured
		self.divided_img = np.zeros((self.w + 2, self.h + 2), dtype=int)

		# Add a 1 pixel border all around the image with ID -1
		self.divided_img[0, :] = -1
		self.divided_img[-1, :] = -1
		self.divided_img[:, 0] = -1
		self.divided_img[:, -1] = -1

	def initialize_weights(self, weights_vert, weights_hori, weights_vert_hard, weights_hori_hard):
		"""
		Sets the values of all weights between pixels
		:param weights_vert: soft vertical weights between pixels
		:param weights_hori: soft horizontal weights between pixels
		:param weights_vert_hard: hard vertical weights between pixels
		:param weights_hori_hard: hard horizontal weights between pixels
		"""
		self.weights_vert = weights_vert
		self.weights_hori = weights_hori
		self.weights_vert_hard = weights_vert_hard
		self.weights_hori_hard = weights_hori_hard

	def initialize_seeds(self):
		"""
		Randomly places superpixel seeds in the image, subdividing it in regions to have a better distribution of seeds
		"""

		# Number of superpixel seeds per pixel-squared in the image
		density = self.nb_superpixels / (self.w * self.h)

		# Divide the image into regions
		n_regions_w = int(math.ceil(self.w / self.subdivide_size))
		n_regions_h = int(math.ceil(self.h / self.subdivide_size))

		# Last regions can have different sizes, treat them aside
		last_region_w = self.w % self.subdivide_size
		last_region_h = self.h % self.subdivide_size
		if last_region_w == 0:
			last_region_w = self.subdivide_size
		if last_region_h == 0:
			last_region_h = self.subdivide_size
		n_seeds = 0

		# Loop on each region subdividing the image
		for region_x in range(n_regions_w):

			# width, starting x and ending x of the region
			region_w = self.subdivide_size if region_x < n_regions_w - 1 else last_region_w
			region_start_x = region_x * self.subdivide_size + 1
			region_end_x = region_start_x + region_w

			for region_y in range(n_regions_h):

				# height, starting y and ending y of the region
				region_h = self.subdivide_size if region_y < n_regions_h - 1 else last_region_h
				region_start_y = region_y * self.subdivide_size + 1
				region_end_y = region_start_y + region_h

				# Use density and region's area to compute the number of seeds to create here
				area = region_w * region_h
				region_seeds = int(math.ceil(area * density))

				# Place seeds randomly in the region, but avoids edges (heuristic based on between pixels weights)
				for _ in range(region_seeds):
					seed_x = seed_y = None
					if seed_x is None or self.weights_vert[seed_x, seed_y] * self.weights_hori[seed_x, seed_y] < 0.5:
						seed_x = random.randint(region_start_x, region_end_x - 1)
						seed_y = random.randint(region_start_y, region_end_y - 1)

					seed_id = n_seeds + 1

					superpixel = SuperPixel(seed_id)

					# Define the plotting color of the superpixel as its seed's color
					self.colors.append(self.source_img[seed_x - 1, seed_y - 1])

					# Capture the seed
					self.divided_img[seed_x, seed_y] = seed_id

					# Add the seed's neighbours to the superpixel's neighbourhood
					superpixel.add_neighbours(seed_x, seed_y, self.divided_img, self)

					# Add this superpixel to the list of superpixels
					self.superpixels.append(superpixel)

					n_seeds += 1

	def grow_superpixels(self, verbose=True):
		"""
		Grow superpixels by capturing neighbours iteratively, based on between pixels weights
		:param verbose: False to run silently, True to print progress
		"""
		it = 0
		total = self.h * self.w

		# Iterate until every pixel belongs to a superpixel
		while np.count_nonzero(self.divided_img) != (self.h + 2) * (self.w + 2):
			it += 1

			# Let each superpixel grow by capturing a part of its neighbourhood, based on their capture probabilities
			for sp in self.superpixels:
				current_neighbours, current_weights = sp.get_weighted_neigbours(self.divided_img, self)

				# Can't grow superpixel if no more neighbours
				if current_neighbours is None:
					continue

				# The chance to capture each neighbour is equal to this neighbour's weight
				sample = np.random.random(len(current_neighbours))
				for ni in range(len(current_neighbours)):
					if sample[ni] <= current_weights[ni]:
						sp.capture_pixel(current_neighbours[ni], self.divided_img, self)

			# Printing progress if verbose
			if verbose:
				current = np.count_nonzero(self.divided_img) - self.h * 2 - self.w * 2
				print("\rSuper-Pixelisation: {:.2f}%".format(current / total * 100), end="")
		if verbose:
			print("")

	def get_weights(self, x1, y1, x2, y2):
		"""
		:param x1: x-coordinate of first pixel
		:param y1: y-coordinate of first pixel
		:param x2: x-coordinate of second pixel
		:param y2: y-coordinate of second pixel
		:return: (hard, soft) weights between pixel (x1, y1) and pixel (x2, y2)
		"""
		if x1 == x2:
			y = min(y1, y2)
			return self.weights_vert_hard[x1 - 1, y - 1], self.weights_vert[x1 - 1, y - 1]
		else:
			x = min(x1, x2)
			return self.weights_hori_hard[x - 1, y1 - 1], self.weights_hori[x - 1, y1 - 1]

	def create_karger_graph(self, w_if, w_ib):
		"""
		Creates a graph for the Karger algorithm, from grown superpixels
		:param w_if: terminal weights from source to pixel (foreground)
		:param w_ib: terminal weights from pixel to target (background)
		:return: Karger graph ready for min-cut
		"""
		g = seg.Karger()

		# Add nodes
		g.add_node(0)	# Source
		g.add_node(1)	# Target
		for sp in self.superpixels:
			g.add_node(sp.sid + 1)	# Superpixel

		# Follow the frontier of each superpixel to compute the weight of its edge with neighbouring superpixels
		for sp1 in self.superpixels:
			i1 = sp1.sid

			# Iterate over captured frontier
			for nei, ws in sp1.captured_neighbours.items():
				i2 = self.divided_img[nei]

				# We want to count weights only once, so we break the symmetry
				if i1 >= i2:
					continue

				# Add the sum of pixel's weights to the edge (it accumulates without overwriting, and works in both edge's directions)
				g.add_edge(i1 + 1, i2 + 1, sum(ws))

		# Compute terminal weights of superpixels
		merge_source = set()	# List of superpixels to merge with the source node directly
		merge_target = set()	# List of superpixels to merge with the target node directly
		for x in range(self.w):
			for y in range(self.h):
				i = self.divided_img[x + 1, y + 1]

				# Accumulate terminal weights for a given pixel and the superpixel it belongs to
				g.add_edge(i + 1, 0, w_if[x, y])
				g.add_edge(i + 1, 1, w_ib[x, y])

				# If weight is above this threshold, it is considered as a part of source or target directly (= scribbles from user)
				if w_if[x, y] > 5000:
					merge_source.add(i + 1)
					self.colors[i - 1] = np.array([0, 0, 255])	# Changes the color of superpixel for plotting
				elif w_ib[x, y] > 5000:
					merge_target.add(i + 1)
					self.colors[i - 1] = np.array([255, 0, 0])	# Changes the color of superpixel for plotting

		# Merge necessary nodes with source
		for node in merge_source:
			g.contract_edge(0, node)

		# Merge necessary nodes with target
		for node in merge_target:
			g.contract_edge(1, node)

		return g

	def get_labeled_image(self, labels):
		"""
		Computes a red/blue image based on superpixel labels
		:param labels: tuple of lists of node indices, [0] being the foreground nodes, [1] being the background nodes
		:return: segmented image of the same shape as the source image, with blue where foreground is and red where background is
		"""

		# Source - updating superpixel's color
		for src_lbl in labels[0]:
			if src_lbl >= 2:
				self.colors[src_lbl - 2] = np.array([0, 0, 255])

		# Target - updating superpixel's color
		for tar_lbl in labels[1]:
			if tar_lbl >= 2:
				self.colors[tar_lbl - 2] = np.array([255, 0, 0])

		# Color new image (per pixel)
		new_img = np.zeros_like(self.source_img)
		for x in range(self.w):
			for y in range(self.h):
				new_img[x, y, :] = self.colors[self.divided_img[x + 1, y + 1] - 1]

		return new_img

	def plot(self, pause=False):
		"""
		Plot the image made of superpixels
		:param pause: True = only displays for 0.01 seconds, False = blocks execution
		"""
		# Attribute to each pixel the color of its associated superpixel (seed color)
		new_img = np.zeros_like(self.source_img)
		for x in range(self.w):
			for y in range(self.h):
				new_img[x, y, :] = self.colors[self.divided_img[x + 1, y + 1] - 1]

		# Color frontier pixels in black to display the edge between superpixels
		for sp in self.superpixels:
			for nei in sp.captured_neighbours.keys():
				new_img[nei[0] - 1, nei[1] - 1] = (0, 0, 0)

		# Plot in superposition of source image, to compare edges
		plt.imshow(self.source_img.swapaxes(0, 1))
		plt.imshow(new_img.swapaxes(0, 1).astype(np.uint8), alpha=0.8)
		if pause:
			plt.pause(0.01)
		else:
			plt.show()





