import image_segmentation as seg
import maxflow
import numpy as np

def perform_image_segmentation(image, scribbles):
	algorithm = gui.algorithm

	weights = seg.GmmWeights(non_terminal_sigma=20, terminal_lambda=0.1, mixture_components=4)
	weights.compute_weights(image, scribbles)

	result = np.zeros_like(image)
	if algorithm == 'maxflow':
		graph, nodes = weights.build_maxflow_graph()
		graph.maxflow()
		result = weights.build_image_from_maxflow_labels(graph, nodes)
	elif algorithm == 'boykov-kolmogorov':
		graph = seg.BoykovKolmogorov(image.shape[0], image.shape[1], weights.w_if, weights.w_ib, weights.hori_w_ij, weights.vert_w_ij)
		graph.do_cut()
		result = graph.get_labeled_image()
	elif algorithm == 'push-relabel':
		pass
	elif algorithm == 'karger':
		superpixeliser = seg.SuperPixeliser(image, nb_superpixels=1000, subdivide_size=100)
		superpixeliser.initialize_weights(weights.vert_w_ij, weights.hori_w_ij, weights.vert_w_hard, weights.hori_w_hard)
		superpixeliser.initialize_seeds()
		superpixeliser.grow_superpixels(verbose=True)
		superpixeliser.plot()
		graph = superpixeliser.create_karger_graph(weights.w_if, weights.w_ib)
		best_cut, best_labels = graph.perform_karger(1000)
		result = superpixeliser.get_labeled_image(best_labels)

	return result


gui = seg.Gui(segmentation_function=perform_image_segmentation)
gui.start()

