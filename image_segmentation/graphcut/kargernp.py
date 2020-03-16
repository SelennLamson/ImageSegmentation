import re
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time

SOURCE = 0
TARGET = 1
TOP = 2
RIGHT = 3
BOTTOM = 4
LEFT = 5
TOTAL = 6

class KargerNp:
	def __init__(self, width, height):
		self.w = width
		self.h = height
		self.all_pixels = np.arange(self.w * self.h, dtype=int)
		self.px_edges = np.arange(6)
		self.labels = (self.all_pixels + 2).reshape(self.w, self.h)
		self.weights = np.zeros((self.w, self.h, 7), dtype=float)

		self.remaining_labels = set(self.all_pixels + 2)

		for x in range(self.w):
			for y in range(self.h):
				assert self.labels[x, y] in self.remaining_labels

	def __len__(self):
		return len(self.remaining_labels)

	def __repr__(self):
		result = "--- Graph: ---\n"
		result += self.labels + "\n"
		result += "-------------"
		return result

	def copy(self):
		new_graph = KargerNp(self.w, self.h)
		new_graph.labels = self.labels.copy()
		new_graph.weights = self.weights.copy()
		new_graph.remaining_labels = self.remaining_labels.copy()
		return new_graph

	def set_terminal_weights(self, w_source, w_target):
		self.weights[:, :, SOURCE] = w_source.copy()
		self.weights[:, :, TARGET] = w_target.copy()

		for x in range(self.w):
			for y in range(self.h):
				if w_source[x, y] > 5000:
					self.remaining_labels.remove(self.labels[x, y])
					self.contaminate_st(x, y, SOURCE)
				elif w_target[x, y] > 5000:
					self.remaining_labels.remove(self.labels[x, y])
					self.contaminate_st(x, y, TARGET)

	def set_horizontal_weights(self, hori_weights):
		self.weights[:-1, :, RIGHT] = hori_weights.copy()
		self.weights[1:, :, LEFT] = hori_weights.copy()

	def set_vertical_weights(self, vert_weights):
		self.weights[:, :-1, BOTTOM] = vert_weights.copy()
		self.weights[:, 1:, TOP] = vert_weights.copy()

	def compute_totals(self):
		self.weights[:, :, TOTAL] = np.sum(self.weights[:, :, :TOTAL], axis=2)

	def get_random_edge(self):
		total_weights = self.weights[:, :, TOTAL].reshape(self.w * self.h)
		total_weights = total_weights / total_weights.sum()

		px1 = np.random.choice(self.all_pixels, 1, p=total_weights)[0]
		x = px1 // self.h
		y = px1 % self.h

		assert total_weights[px1] > 0

		edge = np.random.choice(self.px_edges, 1, p=self.weights[x, y, :TOTAL] / self.weights[x, y, TOTAL])[0]

		assert self.weights[x, y, edge] > 0

		return x, y, edge

	def remove_edge(self, x, y, edge):
		self.weights[x, y, TOTAL] = max(self.weights[x, y, TOTAL] - self.weights[x, y, edge], 0)
		self.weights[x, y, edge] = 0

	def contaminate_st(self, x, y, new_label):
		old_label = self.labels[x, y]

		self.labels[x, y] = new_label
		self.remove_edge(x, y, SOURCE)
		self.remove_edge(x, y, TARGET)

		if x < self.w - 1 and self.labels[x + 1, y] < 2:
			self.remove_edge(x, y, RIGHT)
			self.remove_edge(x + 1, y, LEFT)
		elif x < self.w - 1 and self.labels[x + 1, y] == old_label:
			self.remove_edge(x, y, RIGHT)
			self.remove_edge(x + 1, y, LEFT)
			self.contaminate_st(x + 1, y, new_label)

		if x > 0 and self.labels[x - 1, y] < 2:
			self.remove_edge(x, y, LEFT)
			self.remove_edge(x - 1, y, RIGHT)
		elif x > 0 and self.labels[x - 1, y] == old_label:
			self.remove_edge(x, y, LEFT)
			self.remove_edge(x - 1, y, RIGHT)
			self.contaminate_st(x - 1, y, new_label)

		if y < self.h - 1 and self.labels[x, y + 1] < 2:
			self.remove_edge(x, y, BOTTOM)
			self.remove_edge(x, y + 1, TOP)
		elif y < self.h - 1 and self.labels[x, y + 1] == old_label:
			self.remove_edge(x, y, BOTTOM)
			self.remove_edge(x, y + 1, TOP)
			self.contaminate_st(x, y + 1, new_label)

		if y > 0 and self.labels[x, y - 1] < 2:
			self.remove_edge(x, y, TOP)
			self.remove_edge(x, y - 1, BOTTOM)
		elif y > 0 and self.labels[x, y - 1] == old_label:
			self.remove_edge(x, y, TOP)
			self.remove_edge(x, y - 1, BOTTOM)
			self.contaminate_st(x, y - 1, new_label)

	def contaminate(self, x, y, new_label):
		old_label = self.labels[x, y]

		self.labels[x, y] = new_label

		if x < self.w - 1 and self.labels[x + 1, y] == new_label:
			self.remove_edge(x, y, RIGHT)
			self.remove_edge(x + 1, y, LEFT)
		elif x < self.w - 1 and self.labels[x + 1, y] == old_label:
			self.contaminate(x + 1, y, new_label)

		if x > 0 and self.labels[x - 1, y] == new_label:
			self.remove_edge(x, y, LEFT)
			self.remove_edge(x - 1, y, RIGHT)
		elif x > 0 and self.labels[x - 1, y] == old_label:
			self.contaminate(x - 1, y, new_label)

		if y < self.h - 1 and self.labels[x, y + 1] == new_label:
			self.remove_edge(x, y, BOTTOM)
			self.remove_edge(x, y + 1, TOP)
		elif y < self.h - 1 and self.labels[x, y + 1] == old_label:
			self.contaminate(x, y + 1, new_label)

		if y > 0 and self.labels[x, y - 1] == new_label:
			self.remove_edge(x, y, TOP)
			self.remove_edge(x, y - 1, BOTTOM)
		elif y > 0 and self.labels[x, y - 1] == old_label:
			self.contaminate(x, y - 1, new_label)

	def contract_edge(self, x, y, edge):
		for ox in range(self.w):
			for oy in range(self.h):
				assert self.labels[ox, oy] in self.remaining_labels or self.labels[ox, oy] < 2


		if edge == SOURCE or edge == TARGET:
			self.remove_edge(x, y, SOURCE)
			self.remove_edge(x, y, TARGET)

			self.remaining_labels.remove(self.labels[x, y])
			self.contaminate_st(x, y, edge)

		else:
			nx, ny = x, y
			if edge == TOP:
				ny -= 1
			elif edge == RIGHT:
				nx += 1
			elif edge == BOTTOM:
				ny += 1
			elif edge == LEFT:
				nx -= 1

			lbl1 = self.labels[x, y]
			lbl2 = self.labels[nx, ny]

			if lbl1 < 2 and lbl2 < 2:
				return

			# Inverting to get lbl2 as source or target and only code this case once
			if lbl1 < 2:
				x, y, nx, ny = nx, ny, x, y
				lbl1, lbl2 = lbl2, lbl1

			# Case when the pixel we are merging from is SOURCE or TARGET (or reversed situation)
			if lbl2 < 2:
				self.remaining_labels.remove(lbl1)
				self.contaminate_st(x, y, lbl2)

			# Case where we are merging from another normal pixel
			else:
				self.remaining_labels.remove(lbl1)
				self.contaminate(x, y, lbl2)

	def perform_random_cut(self):
		h = self.copy()
		while len(h) > 0:
			h.contract_edge(*h.get_random_edge())
		return h

	def perform_karger(self, n_iter):
		best_cut = None
		best_labels = None
		for it in range(n_iter):
			h = self.perform_random_cut()
			cut = h.weights.sum()

			if best_cut is None or cut < best_cut:
				best_cut = cut
				best_labels = h.labels.copy()

			print("\rKarger: {:.2f}%, best-cut: {}".format(100 * it / n_iter, best_cut), end="")
		print("")
		print('best labels: ', best_labels)
		return best_cut, best_labels

