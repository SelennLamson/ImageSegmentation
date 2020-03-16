import image_segmentation as seg
import matplotlib.pyplot as plt

def perform_image_segmentation(image, scribbles):
    """
    :param image: image that we want to segment
    :param scribbles: scribbles made using pygame (gui.py) delimiting background from foreground
    """
    # Use Weights Class to compute the terminal and non terminal edges of the graph
    weights = seg.Weights(non_terminal_sigma=10, terminal_lambda=1)
    weights.compute_weights(image, scribbles)

    # Create Superpixels class to transform the image
    # Consider superpixels grown cleverly around uniformly distributed seeds, instead of simple pixels
    superpixeliser = seg.SuperPixeliser(image, nb_superpixels=100, subdivide_size=100)
    superpixeliser.initialize_weights(weights.vert_w_ij, weights.hori_w_ij, weights.vert_w_hard, weights.hori_w_hard)
    superpixeliser.initialize_seeds()
    superpixeliser.grow_superpixels(verbose=True)

    # Create the graph corresponding to above image transformation
    graph = superpixeliser.create_karger_graph(weights.w_if, weights.w_ib)
    superpixeliser.plot()
    #result = graph

    # Apply Karger (n_times iterations) to find best cut
    n_times = 1
    # best_cut, best_labels = graph.perform_karger(n_times)

    # Apply Alex Karger
    # karger = seg.Karger()
    # best_cut, best_labels = karger.karger_mincut(graph, n_times)
    
    # Apply push relabel
    seg.solve_max_flow(graph, 0, 1)

    # Return image where each pixel is colored as foreground (blue) or background (red)
    result = superpixeliser.get_labeled_image(best_labels)
    print(type(result))
    plt.imshow(result)

    return result

# Apply
gui = seg.Gui(segmentation_function=perform_image_segmentation)
gui.start()










