import image_segmentation as seg


def perform_image_segmentation(image, scribbles):
	weights = seg.Weights(non_terminal_sigma=10, terminal_lambda=10)
	weights.compute_weights(image, scribbles)

	superpixeliser = seg.SuperPixeliser(image, nb_superpixels=10000, subdivide_size=100)
	superpixeliser.initialize_weights(weights.vert_w_hard, weights.hori_w_hard)

	superpixeliser.initialize_seeds()
	superpixeliser.grow_superpixels(verbose=True)
	superpixeliser.plot()
	superpixeliser.create_graph(weights.vert_w_ij, weights.hori_w_ij, weights.w_if, weights.w_ib)


gui = seg.Gui(segmentation_function=perform_image_segmentation)
gui.start()

