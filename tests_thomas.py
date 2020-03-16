import image_segmentation as seg
import maxflow
from image_segmentation.graphcut.boykov_kolmogorov_np import BoykovKolmogorov


def perform_image_segmentation(image, scribbles):
	weights = seg.GmmWeights(non_terminal_sigma=20, terminal_lambda=.1, mixture_components=3)
	weights.compute_weights(image, scribbles)

	graph = BoykovKolmogorov(image.shape[0], image.shape[1], weights.w_if, weights.w_ib, weights.hori_w_ij, weights.vert_w_ij)
	graph.do_cut()
	result = graph.get_labeled_image()


	# graph, nodes = weights.build_maxflow_graph()
	# graph.maxflow()
	# result = weights.build_image_from_maxflow_labels(graph, nodes)

	# superpixeliser = seg.SuperPixeliser(image, nb_superpixels=10000, subdivide_size=100)
	# superpixeliser.initialize_weights(weights.vert_w_ij, weights.hori_w_ij, weights.vert_w_hard, weights.hori_w_hard)
	#
	# superpixeliser.initialize_seeds()
	# superpixeliser.grow_superpixels(verbose=True)
	# # graph = superpixeliser.create_graph(weights.w_if, weights.w_ib)
	# graph, nodes = superpixeliser.create_maxflow_graph(weights.w_if, weights.w_ib)
	# superpixeliser.plot()
	#
	# flow = graph.maxflow()
	# result = superpixeliser.get_labeled_image_maxflow(graph, nodes)

	# best_cut, best_labels = graph.perform_karger(10000)
	# result = superpixeliser.get_labeled_image(best_labels)

	return result


gui = seg.Gui(segmentation_function=perform_image_segmentation)
gui.start()

