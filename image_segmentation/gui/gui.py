import sys
import os
import pygame
import pickle
import numpy as np

from image_segmentation.utils import *


DRAW_RADIUS = 3


class Gui:
	def __init__(self, segmentation_function):
		self.segmentation_function = segmentation_function

		self.source_image = None
		self.scribbles = None
		self.results = None
		self.image_size = None

		self.resized_image = None
		self.resized_scribbles = None
		self.resized_results = None

		self.screen = None
		self.screen_size = None
		self.font = None

		self.image_position = (0, 0)
		self.image_zoom = 1

		self.prev_draw_pos = (0, 0)
		self.prev_draw_mode = 0

		self.algorithms = ['maxflow', 'boykov-kolmogorov', 'push-relabel', 'karger']
		self.algorithm = self.algorithms[0]


	def update_screen(self, size_changed, draw_changed):
		im_w, im_h = self.image_size
		sc_w, sc_h = self.screen_size

		w_ratio = im_w / sc_w
		h_ratio = im_h / sc_h
		self.image_zoom = max(w_ratio, h_ratio)

		new_w = int(im_w / self.image_zoom)
		new_h = int(im_h / self.image_zoom)

		self.image_position = ((sc_w - new_w) // 2, (sc_h - new_h) // 2)

		if size_changed:
			self.resized_image = pygame.transform.scale(self.source_image, (new_w, new_h))
			self.resized_results = pygame.transform.scale(self.results, (new_w, new_h))

		if size_changed or draw_changed:
			self.resized_scribbles = pygame.transform.scale(self.scribbles, (new_w, new_h))
			self.screen.fill((0, 0, 0))
			self.screen.blit(self.resized_image, self.image_position)
			self.screen.blit(self.resized_results, self.image_position)
			self.screen.blit(self.resized_scribbles, self.image_position)

			alg_txt = self.font.render("Algorithm: " + self.algorithm, True, (255, 255, 255))
			self.screen.blit(alg_txt, (10, 10))


	def start(self, file_name=None):
		# --- LOADING A SAVED FILE ---
		# Define folder where images are stored
		base_path = "images/"
		if file_name is None or not os.path.exists(base_path + file_name):
			saved = [f for f in os.listdir(base_path) if f.endswith('.png') or f.endswith('.png') or f.endswith('.jpeg')]
			if len(saved) == 1:
				ans = 0
			else:
				while True:
					for i, s in enumerate(saved):
						print('- [' + str(i) + '] ' + s)
					ans = input("Chose an image: ")
					try:
						ans = int(ans)
						assert (0 <= ans < len(saved))
						break
					except (ValueError, AssertionError):
						pass

			# Define filename as the desired image's name (e.g 'polar_bear.png')
			file_name = saved[ans]

		# Load image in pygame
		self.source_image = pygame.image.load(base_path + file_name)
		# Store image size
		self.image_size = (self.source_image.get_width(), self.source_image.get_height())


		# --- INITIALIZING PYGAME ---
		pygame.init()
		screen_info = pygame.display.Info()
		self.screen_size = (screen_info.current_w // 2, screen_info.current_h // 2)

		self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
		clock = pygame.time.Clock()
		self.font = pygame.font.SysFont('consolas', 20, True)

		# --- LOADING ASSETS ---
		self.scribbles = pygame.Surface(self.image_size, pygame.SRCALPHA)
		self.scribbles.fill((0, 0, 0, 0))
		self.results = pygame.Surface(self.image_size, pygame.SRCALPHA)
		self.results.fill((0, 0, 0, 0))


		# --- MAIN LOOP ---
		while 1:
			size_changed = False
			draw_changed = False

			# --- EVENTS ---
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()
				if event.type == pygame.KEYDOWN:
					if event.key in [pygame.K_KP_ENTER, pygame.K_RETURN]:
						np_image = pygame.surfarray.array3d(self.source_image)
						np_scribbles = pygame.surfarray.array3d(self.scribbles)

						#pygame.quit()

						# Apply segmentation function
						np_results = self.segmentation_function(np_image, np_scribbles)

						# Use results from segmentation method above
						rgb_results = pygame.surfarray.pixels3d(self.results)
						alpha_results = pygame.surfarray.pixels_alpha(self.results)
						rgb_results[:, :, :] = np_results
						alpha_results[:, :] = (np.sum(np_results, axis=2) > 0) * 128
						del rgb_results
						del alpha_results

						size_changed = True
						draw_changed = True
					if event.key == pygame.K_r:
						self.scribbles = pygame.Surface(self.image_size, pygame.SRCALPHA)
						self.scribbles.fill((0, 0, 0, 0))
						self.results = pygame.Surface(self.image_size, pygame.SRCALPHA)
						self.results.fill((0, 0, 0, 0))
						size_changed = True
					if event.key == pygame.K_1 and len(self.algorithms) >= 1:
						self.algorithm = self.algorithms[0]
						draw_changed = True
					if event.key == pygame.K_2 and len(self.algorithms) >= 2:
						self.algorithm = self.algorithms[1]
						draw_changed = True
					if event.key == pygame.K_3 and len(self.algorithms) >= 3:
						self.algorithm = self.algorithms[2]
						draw_changed = True
					if event.key == pygame.K_4 and len(self.algorithms) >= 4:
						self.algorithm = self.algorithms[3]
						draw_changed = True
					if event.key == pygame.K_5 and len(self.algorithms) >= 5:
						self.algorithm = self.algorithms[4]
						draw_changed = True

				if event.type == pygame.VIDEORESIZE:
					self.screen_size = (event.w, event.h)
					self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
					size_changed = True


			# --- DRAWING ---
			cursor_x, cursor_y = pygame.mouse.get_pos()
			cursor_x = int((cursor_x - self.image_position[0]) * self.image_zoom)
			cursor_y = int((cursor_y - self.image_position[1]) * self.image_zoom)

			draw_mode = 0
			if pygame.mouse.get_pressed()[0]:
				draw_mode = 1
			elif pygame.mouse.get_pressed()[2]:
				draw_mode = 2

			if draw_mode > 0:
				draw_changed = True

				pygame.draw.line(self.scribbles,
								 FOREGROUND_RGBA if draw_mode == 1 else BACKGROUND_RGBA,
								 self.prev_draw_pos,
								 (cursor_x, cursor_y),
								 DRAW_RADIUS * 2)

				pygame.draw.circle(self.scribbles,
								   FOREGROUND_RGBA if draw_mode == 1 else BACKGROUND_RGBA,
								   (cursor_x, cursor_y),
								   DRAW_RADIUS - 1)

			self.prev_draw_pos = (cursor_x, cursor_y)
			self.prev_draw_mode = draw_changed


			# --- UPDATING SCREEN ---
			self.update_screen(size_changed, draw_changed)
			pygame.display.flip()
			clock.tick(120)
