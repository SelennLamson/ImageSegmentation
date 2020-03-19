import image_segmentation as seg
import matplotlib.pyplot as plt

def perform_image_segmentation(image, scribbles):
    """
    :param image: image that we want to segment
    :param scribbles: scribbles made using pygame (gui.py) delimiting background from foreground
    Includes the whole process of the image segmentation. 
    Can use different algorithm alhtough the code is for Karger here
    """
    # Use Weights Class to compute the terminal and non terminal edges of the graph
    weights = seg.Weights(non_terminal_sigma=10, terminal_lambda=1)
    weights.compute_weights(image, scribbles)

    # PUSH RELABEL 
    # push_relabel = seg.PushRelabel()
    # source = 0  # A
    # sink = weights.w_if.shape[1] * weights.w_if.shape[0] + 1 # F
    # push_relabel.capacity_matrix(weights.w_if, weights.w_ib, weights.hori_w_ij, weights.vert_w_ij)
    # push_relabel.MaxFlow(push_relabel.capacity, source, sink)

    # Create SUPERPIXELS class to transform the image
    # Consider superpixels grown cleverly around uniformly distributed seeds, instead of simple pixels
    superpixeliser = seg.SuperPixeliser(image, nb_superpixels=500, subdivide_size=100)
    superpixeliser.initialize_weights(weights.vert_w_ij, weights.hori_w_ij, weights.vert_w_hard, weights.hori_w_hard)
    superpixeliser.initialize_seeds()
    superpixeliser.grow_superpixels(verbose=True)

    # Create the graph corresponding to above image transformation
    graph = superpixeliser.create_karger_graph(weights.w_if, weights.w_ib)
    superpixeliser.plot()

    # Apply KARGER (n_times iterations) to find best cut
    n_times = 30000
    best_cut, best_labels = graph.perform_karger(n_times)

    # Apply NETWORKX KARGER
    # karger = seg.Karger()
    # best_cut, best_labels = karger.karger_mincut(graph, n_times)

    # Return image where each pixel is colored as foreground (blue) or background (red)
    result = superpixeliser.get_labeled_image(best_labels)
    print(type(result))
    plt.imshow(result)

    return result

# Apply this function within GUI framework (for scribbles)
gui = seg.Gui(segmentation_function=perform_image_segmentation)
gui.start()


