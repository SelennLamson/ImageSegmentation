import image_segmentation as seg
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

FOREGROUND = (0, 0, 255)
BACKGROUND = (255, 0, 0)

def gaussian(x, mu, sig):
    return np.exp(-(x - mu)**2 / (2 * sig**2))

class Weights:
    def __init__(self, non_terminal_sigma=1, terminal_lambda=10):
        self.non_terminal_sigma = non_terminal_sigma
        self.terminal_lambda = terminal_lambda

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
        :return: two dicts:
        - mu_dict:  key is the scribble color, value is the mean of every pixels under the scribbles of yuv image, shape (1,3)
        - sigma_dict:  key is the scribble color, value is a (3 x 3) symetric matrix,
        [ var(y) cov(yu) cov(yv) ]
        [ cov(uy) var(u) cov(uv) ]
        [ cov(vy) cov(vu) var(v) ]
        """

        scribbles = self.find_scribbles(scribl_rgb)
        comps = defaultdict(lambda: np.array([]).reshape(0, 3))

        for (i, j) in scribbles:
            c = tuple(scribl_rgb[i, j, :])
            comps[c] = np.vstack([comps[c], img_yuv[i, j, :]])

        mu, Sigma = {}, {}
        for c in comps:
            mu[c] = np.mean(comps[c], axis=0)
            Sigma[c] = np.cov(comps[c].T)

        # Plotting Y
        channel = 0
        values_f = comps[FOREGROUND][:, channel]
        values_b = comps[BACKGROUND][:, channel]

        gaussian_f = gaussian(np.linspace(0, 255, 255), mu[FOREGROUND][channel], Sigma[FOREGROUND][channel, channel])
        gaussian_b = gaussian(np.linspace(0, 255, 255), mu[BACKGROUND][channel], Sigma[BACKGROUND][channel, channel])

        # fig, axs = plt.subplots(1, 2, sharex=True)
        # axs[0].hist(values_f, density=True, color="#0000ff88")
        # axs[0].hist(values_b, density=True, color="#ff000088")
        # axs[1].plot(gaussian_f, color="#0000ff")
        # axs[1].plot(gaussian_b, color="#ff0000")

        # plt.plot(gaussian_f, color="#0000ff")
        # plt.plot(gaussian_b, color="#ff0000")
        plt.show()

        return mu, Sigma

    def non_terminal_weights(self, matrix):
        """
        :param matrix: matrix to apply the function
        :return: the weight of the edge between two pixels.
        weight is large if pixels are similar and low if not
        """
        return np.exp((-1 / (2 * self.non_terminal_sigma ** 2)) * matrix)

    def terminal_color_proba(self, val, mu, sig, image_group):
        """
        :param val: pixel (y, u, v)
        :param mu: dictionnary containing the mean value of pixels y u v
        :param sig: dictionnary containing the covariance matrix
        :param image_group: F for foreground and B for background
        :return: the probability of being the color of the pixel value while being of forground or background
        """
        two_pi_k = (2 * np.pi) ** 3
        # value = np.linalg.norm(val)
        if image_group == 'F':
            mean = mu[(0, 0, 255)]
            sigma = sig[(0, 0, 255)]

            diff = val - mean
            return np.exp(-0.5 * diff.T @ np.linalg.inv(sigma) @ diff)\
                   / np.sqrt(two_pi_k * np.linalg.det(sigma))

        elif image_group == 'B':

            mean = mu[(255, 0, 0)]
            sigma = sig[(255, 0, 0)]

            diff = val - mean
            return np.exp(-0.5 * diff.T @ np.linalg.inv(sigma) @ diff)\
                   / np.sqrt(two_pi_k * np.linalg.det(sigma))

    def terminal_class_proba(self, img_yuv, group, mu, Sigma):
        """
        :param img_yuv: image values (YUV format)
        :param group: BACKGROUND or FOREGROUND constant
        :param mu: dictionnary containing the mean value of pixels y u v
        :param Sigma: dictionnary containing the covariance matrix
        """
        two_pi_k = (2 * np.pi) ** 3
        mean = mu[group]
        sigma = Sigma[group]
        diff = img_yuv - mean[np.newaxis, np.newaxis, :]
        res = np.matmul(diff, np.linalg.inv(sigma)[np.newaxis, np.newaxis, :])[0]
        res = np.sum(res * diff, axis=2)
        return np.exp(-0.5 * res) / np.sqrt(two_pi_k * np.linalg.det(sigma))

    def compute_weights(self, img_rgb, scribl_rgb):
        """
        :param img_rgb: numpy array of shape (w, h, 3) containing the RGB image to work on
        :param scribl_rgb: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :return: a numpy array of shape (w, h, 3) with blue and red everywhere, after segmentation
        """
        # Create YUV arrays
        # Rename arrays to their according color mask
        img_rgb = cv2.bilateralFilter(img_rgb, 15, 20, 20)
        # plt.imshow(img_rgb.swapaxes(0, 1))
        # plt.show()

        # img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_yuv = img_rgb

        mu, Sigma = self.get_probab_param(img_yuv, scribl_rgb)

        # Compute non-terminal edge weights
        # Initialize zeros vectors to deal with edges
        vert_norm = np.linalg.norm(img_yuv[:, 1:] - img_yuv[:, :-1], axis=2)
        hori_norm = np.linalg.norm(img_yuv[1:, :] - img_yuv[:-1, :], axis=2)

        self.vert_w_ij = self.non_terminal_weights(vert_norm)
        self.hori_w_ij = self.non_terminal_weights(hori_norm)

        # Compute terminal edges weights
        pf = self.terminal_class_proba(img_yuv, FOREGROUND, mu, Sigma)
        pb = self.terminal_class_proba(img_yuv, BACKGROUND, mu, Sigma)
        pbf = pf + pb

        self.w_if = -self.terminal_lambda * np.log10(pb / pbf)
        self.w_ib = -self.terminal_lambda * np.log10(pf / pbf)

        # heatmap = np.zeros_like(vert_w_ij)
        # heatmap[:, :, 2] = w_if * 255
        # heatmap[:, :, 0] = w_ib * 255

        canny = cv2.Canny(img_yuv, 10, 10)
        self.hori_w_hard = (1 - np.max(np.array([canny[1:, :], canny[:-1, :]]), axis=0)/255) * 0.9 + 0.1
        self.vert_w_hard = (1 - np.max(np.array([canny[:, 1:], canny[:, :-1]]), axis=0)/255) * 0.9 + 0.1

        # plt.imshow(self.hori_w_ij.swapaxes(0, 1), cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(self.vert_w_ij.swapaxes(0, 1), cmap='gray', vmin=0, vmax=1)
        # plt.show()



        # print(mu)
        # print('---' * 10)
        # print(Sigma)
        # print('---' * 10)
        # print(w_if)
        # print('---' * 10)
        # print(w_ib)
        # print('---' * 10)
        # print(w_if.shape)
        # print('---' * 10)
        # print(w_ib.shape)
        # print('---' * 10)
        # print(vert_w_ij.shape)
        # print('---' * 10)
        # print(hori_w_ij.shape)
        # print('---' * 10)
        # print(vert_w_ij)
        # print('---' * 10)
        # print(hori_w_ij)

        # return scribl_rgb
