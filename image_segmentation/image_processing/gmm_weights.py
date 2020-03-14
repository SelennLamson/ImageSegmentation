import image_segmentation as seg
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import maxflow
from sklearn.mixture import GaussianMixture

FOREGROUND = (0, 0, 255)
BACKGROUND = (255, 0, 0)

def gaussian(x, mu, sig):
    """
    :param x: data
    :param mu: mean
    :param sig: standard deviation
    :return: pdf of gaussian distribution
    """
    return 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sig**2))

class GmmWeights:
    def __init__(self, non_terminal_sigma=1, terminal_lambda=10, mixture_components=5):
        self.non_terminal_sigma = non_terminal_sigma
        self.terminal_lambda = terminal_lambda
        self.mixture_components = mixture_components

        self.w_if = None
        self.w_ib = None
        self.vert_w_ij = None
        self.hori_w_ij = None
        self.vert_w_hard = None
        self.hori_w_hard = None

    def find_scribbles(self, scribble_img):
        """
        :param scribble_img: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :return: a numpy array with the location of the scribbled pixels
        """
        xy = np.where(np.any(scribble_img != [0, 0, 0], axis=-1))
        return np.array(xy).T

    def get_probab_param(self, img_yuv, scribl_rgb):
        """
        :param img_yuv: numpy array of shape (w, h, 3) containing the YUV image
        :param scribl_rgb: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :return: one models dict
        """
        # Get coordinates of scribbles
        scribbles = self.find_scribbles(scribl_rgb)
        comps = defaultdict(lambda: np.array([]).reshape(0, 3))

        # Separately store background and foreground
        for (i, j) in scribbles:
            c = tuple(scribl_rgb[i, j, :])
            comps[c] = np.vstack([comps[c], img_yuv[i, j, :]])

        # Compute mean pixel value and variance of the scribbles
        # Mixture of Gaussians instead of multivariate gaussian
        gmm = {}
        for c in comps:
            gmm[c] = GaussianMixture(n_components=self.mixture_components)
            gmm[c].fit(comps[c])

        # Plotting Y
        # values_f1 = comps[FOREGROUND][:, 0]
        # values_b1 = comps[BACKGROUND][:, 0]
        # values_f2 = comps[FOREGROUND][:, 1]
        # values_b2 = comps[BACKGROUND][:, 1]
        # values_f3 = comps[FOREGROUND][:, 2]
        # values_b3 = comps[BACKGROUND][:, 2]

        # gaussian_f = gaussian(np.linspace(0, 255, 255), mu[FOREGROUND][channel], Sigma[FOREGROUND][channel, channel])
        # gaussian_b = gaussian(np.linspace(0, 255, 255), mu[BACKGROUND][channel], Sigma[BACKGROUND][channel, channel])

        # fig = plt.figure(figsize=(8, 8))
        # # ax = fig.gca(projection='3d')
        # plt.scatter(values_f1, values_f3, color='blue')
        # plt.scatter(values_b1, values_b3, color='red')
        #
        # X, Y, Z = np.meshgrid(np.linspace(0, 255, 100), np.linspace(0, 255, 100), np.linspace(0, 255, 100))
        # XX = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        # S0 = gmm[list(gmm.keys())[0]].score_samples(XX)
        # S0 = np.mean(S0.reshape((100, 100, 100)), axis=1).reshape(100, 100)
        # S1 = gmm[list(gmm.keys())[1]].score_samples(XX)
        # S1 = np.mean(S1.reshape((100, 100, 100)), axis=1).reshape(100, 100)
        #
        # plt.contour(X[:, 0, :], Z[:, 0, :], S0, np.linspace(S0.min(), S0.max(), 20))
        # plt.contour(X[:, 0, :], Z[:, 0, :], S1, np.linspace(S1.min(), S1.max(), 20))
        #
        # plt.show()

        return scribbles, gmm

    def non_terminal_weights(self, matrix):
        """
        :param matrix: matrix to apply the function
        :return: the weight of the edge between two pixels.
        weight is large if pixels are similar and low if not
        """
        return np.exp((-1 / (2 * self.non_terminal_sigma ** 2)) * matrix)

    def terminal_class_proba(self, img_yuv, group, gmm):
        """
        :param img_yuv: image values (YUV format)
        :param group: BACKGROUND or FOREGROUND constant
        :param mu: dictionnary containing the mean value of pixels y u v
        :param Sigma: dictionnary containing the covariance matrix
        """
        model = gmm[group]
        return model.score_samples(img_yuv.reshape(img_yuv.shape[0] * img_yuv.shape[1], 3)).reshape(img_yuv.shape[0], img_yuv.shape[1])

    def compute_weights(self, img_rgb, scribl_rgb):
        """
        :param img_rgb: numpy array of shape (w, h, 3) containing the RGB image to work on
        :param scribl_rgb: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :return: a numpy array of shape (w, h, 3) with blue and red everywhere, after segmentation
        """
        # Create YUV arrays
        # Rename arrays to their according color mask
        img_rgb = cv2.bilateralFilter(img_rgb, 15, 20, 20)

        # img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_yuv = img_rgb

        scribbles, mu, Sigma = self.get_probab_param(img_yuv, scribl_rgb)

        # Compute non-terminal edge weights
        # Initialize zeros vectors to deal with edges
        vert_norm = np.linalg.norm(img_yuv[:, 1:] - img_yuv[:, :-1], axis=2)
        hori_norm = np.linalg.norm(img_yuv[1:, :] - img_yuv[:-1, :], axis=2)
        self.vert_w_ij = self.non_terminal_weights(vert_norm)
        self.hori_w_ij = self.non_terminal_weights(hori_norm)

        # Compute proba of the color given background/foreground
        pf = self.terminal_class_proba(img_yuv, FOREGROUND, mu, Sigma)
        pb = self.terminal_class_proba(img_yuv, BACKGROUND, mu, Sigma)
        pbf = pf + pb

        # Compute terminal edges weights (non scribbled pixels)
        self.w_if = -self.terminal_lambda * np.log10(pb / pbf)
        self.w_ib = -self.terminal_lambda * np.log10(pf / pbf)

        # Plot
        plt.imshow(self.w_if.swapaxes(0, 1), cmap='gray')
        plt.show()
        plt.imshow(self.w_ib.swapaxes(0, 1), cmap='gray')
        plt.show()

        # Change the terminal weights of scribbled pixel to 0 or infinity
        infinity = 10000
        foreground_scribbles = np.all(scribl_rgb[scribbles[:, 0], scribbles[:, 1]] == FOREGROUND, axis=1)
        background_scribbles = np.all(scribl_rgb[scribbles[:, 0], scribbles[:, 1]] == BACKGROUND, axis=1)
        self.w_if[scribbles[:, 0], scribbles[:, 1]] = foreground_scribbles * infinity \
                                                      + (1 - foreground_scribbles) * self.w_if[scribbles[:, 0], scribbles[:, 1]]
        self.w_if[scribbles[:, 0], scribbles[:, 1]] = (1 - background_scribbles) * self.w_if[scribbles[:, 0], scribbles[:, 1]]
        self.w_ib[scribbles[:, 0], scribbles[:, 1]] = background_scribbles * infinity \
                                                      + (1 - background_scribbles) * self.w_ib[scribbles[:, 0], scribbles[:, 1]]
        self.w_ib[scribbles[:, 0], scribbles[:, 1]] = (1 - foreground_scribbles) * self.w_ib[scribbles[:, 0], scribbles[:, 1]]

        # heatmap = np.zeros_like(vert_w_ij)
        # heatmap[:, :, 2] = w_if * 255
        # heatmap[:, :, 0] = w_ib * 255

        # Add a canny edge detector
        canny = cv2.Canny(img_yuv, 10, 10)
        self.hori_w_hard = (1 - np.max(np.array([canny[1:, :], canny[:-1, :]]), axis=0)/255) * 0.9 + 0.1
        self.vert_w_hard = (1 - np.max(np.array([canny[:, 1:], canny[:, :-1]]), axis=0)/255) * 0.9 + 0.1

        # return scribl_rgb


    def build_maxflow_graph(self):
        """
        @:return: graph representation for the image, to apply mincut/maxflow algo
        """
        # Number of edges to draw (no target and source nodes yet)
        w = self.w_if.shape[0]
        h = self.w_if.shape[1]
        n_nodes = w * h
        n_edges = ((w - 1) * h + w * (h - 1)) * 2
        g = maxflow.Graph[float](n_nodes, n_edges)

        nodes = g.add_nodes(n_nodes)

        # Create nodes and join (grid representation)
        for x in range(w):
            for y in range(h):
                node_id = x * h + y
                g.add_tedge(nodes[node_id], self.w_if[x, y], self.w_ib[x, y])

                # Add horizontal edges
                if x < w - 1:
                    neib_id = (x + 1) * h + y
                    g.add_edge(nodes[node_id], nodes[neib_id], self.hori_w_ij[x, y], self.hori_w_ij[x, y])

                # Add vertical edges
                if y < h - 1:
                    neib_id = x * h + y + 1
                    g.add_edge(nodes[node_id], nodes[neib_id], self.vert_w_ij[x, y], self.vert_w_ij[x, y])

        return g, nodes


    def build_image_from_maxflow_labels(self, g, nodes):
        """
        :param g: graph
        :param nodes: set of nodes
        :return: final image with predicted location of background and foreground
        """
        # Number of edges to draw
        w = self.w_if.shape[0]
        h = self.w_if.shape[1]
        new_img = np.zeros((w, h, 3), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                node_id = x * h + y
                if g.get_segment(nodes[node_id]):
                    new_img[x, y, :] = np.array([255, 0, 0])
                else:
                    new_img[x, y, :] = np.array([0, 0, 255])

        return new_img

