import image_segmentation as seg
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import time


class Weights:
    def __init__(self):
        self.non_terminal_sigma = 1
        self.lam = 10

    def find_scribbles(self, sriblled_img):
        """
        :param sriblled_img: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :return: a numpy array with the location of the scribbled pixels
        """
        x_b = np.where((sriblled_img[:, :, 0] == 0) & (sriblled_img[:, :, 1] == 0) & (sriblled_img[:, :, 2] == 255))[0]
        y_b = np.where((sriblled_img[:, :, 0] == 0) & (sriblled_img[:, :, 1] == 0) & (sriblled_img[:, :, 2] == 255))[1]
        b_scribbles = np.dstack((x_b, y_b))[0]

        x_r = np.where((sriblled_img[:, :, 0] == 255) & (sriblled_img[:, :, 1] == 0) & (sriblled_img[:, :, 2] == 0))[0]
        y_r = np.where((sriblled_img[:, :, 0] == 255) & (sriblled_img[:, :, 1] == 0) & (sriblled_img[:, :, 2] == 0))[1]
        r_scribbles = np.dstack((x_r, y_r))[0]

        return np.vstack((b_scribbles, r_scribbles))

    def get_probab_param(self, img_yuv, scribl_rgb, scribl_yuv):
        """

        :param img_yuv: numpy array of shape (w, h, 3) containing the YUV image
        :param scribl_rgb: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :param scribl_yuv: same as scribl_rgb but in the YUV color.
        :return: two dicts:
        - mu_dict:  key is the scribble color, value is the mean of every pixels under the scribbles of yuv image, shape (1,3)
        - sigma_dict:  key is the scribble color, value is a (3 x 3) symetric matrix,
        [ var(y) cov(yu) cov(yv) ]
        [ cov(uy) var(u) cov(uv) ]
        [ cov(vy) cov(vu) var(v) ]
        """

        scribbles = self.find_scribbles(scribl_rgb)
        imageo = np.zeros(scribl_yuv.shape)

        comps = defaultdict(lambda: np.array([]).reshape(0, 3))

        for (i, j) in scribbles:
            imageo[i, j, :] = scribl_rgb[i, j, :]
            # scribble color as key of comps
            comps[tuple(imageo[i, j, :])] = np.vstack([comps[tuple(imageo[i, j, :])], img_yuv[i, j, :]])

        mu, Sigma = {}, {}
        # compute MLE parameters for Gaussians
        for c in comps:
            mu[c] = np.mean(comps[c], axis=0)
            Sigma[c] = np.cov(comps[c].T)

        return mu, Sigma

    def non_terminal_weights(self, matrix, non_terminal_sigma):
        """

        :param sigma:
        :param matrix: matrix to apply the function
        :return: the weight of the edge between two pixels.
        weight is large if pixels are similar and low if not
        """
        return np.exp((-1 / (2 * non_terminal_sigma ** 2)) * matrix)

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
            mean = mu[(0, 0, 255)].astype(np.int64)
            sigma = sig[(0, 0, 255)].astype(np.int64)

            return np.exp(-0.5 * np.dot(np.dot((val - mean), np.linalg.inv(sigma)), (val - mean))) / np.sqrt(
                two_pi_k * np.linalg.det(sigma))

        elif image_group == 'B':

            mean = mu[(255, 0, 0)].astype(np.int64)
            sigma = sig[(255, 0, 0)].astype(np.int64)

            return np.exp(-0.5 * np.dot(np.dot((val - mean), np.linalg.inv(sigma)), (val - mean))) / np.sqrt(
                two_pi_k * np.linalg.det(sigma))

    def terminal_class_proba(self, value, mu, sig, image_group):
        """

        :param value: pixel (y, u, v)
        :param mu: dictionnary containing the mean value of pixels y u v
        :param sig: dictionnary containing the covariance matrix
        :param image_group: F for foreground and B for background
        :return: return the Pdf of Foreground, or Background
        """
        P_f = self.terminal_color_proba(value, mu, sig, 'F')
        P_b = self.terminal_color_proba(value, mu, sig, 'B')

        if image_group == 'F':
            return P_f / (P_f + P_b)

        elif image_group == 'B':
            return P_b / (P_f + P_b)

    def seg_func_dummy(self, img, scrib):
        """
        :param img: numpy array of shape (w, h, 3) containing the RGB image to work on
        :param scrib: numpy array of shape (w, h, 3) with black everywhere and blue/red where scribbles
        :return: a numpy array of shape (w, h, 3) with blue and red everywhere, after segmentation
        """
        # Create YUV arrays
        # Rename arrays to their according color mask
        img_rgb = img
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        scribl_rgb = scrib
        scribl_yuv = cv2.cvtColor(scribl_rgb, cv2.COLOR_RGB2YUV)

        mu, Sigma = self.get_probab_param(img_yuv, scribl_rgb, scribl_yuv)

        # Compute non-terminal edge weights
        # Initialize zeros vectors to deal with edges

        height, width, color = img_rgb.shape

        bot_n_top_norm = np.linalg.norm(img_rgb[1:, :] - img_rgb[:-1, :], axis=2)
        left_n_right_norm = np.linalg.norm(img_rgb[:, 1:] - img_rgb[:, :-1], axis=2)

        bot_n_topw_ij = self.non_terminal_weights(bot_n_top_norm, non_terminal_sigma=1)
        left_n_rightw_ij = self.non_terminal_weights(left_n_right_norm, non_terminal_sigma=1)

        # Compute terminal edges weights
        W_iF = np.zeros((height, width))
        W_iB = np.zeros((height, width))
        for x in tqdm(range(height)):
            for y in range(width):
                W_iF[x, y] = -self.lam * np.log10(self.terminal_class_proba(img_yuv[x, y], mu, Sigma, 'F'))
                W_iB[x, y] = -self.lam * np.log10(self.terminal_class_proba(img_yuv[x, y], mu, Sigma, 'B'))
        # W_iF = -lam * np.log10(terminal_class_proba(img_yuv, mu, Sigma, 'F'))
        # W_iB = -lam * np.log10(terminal_class_proba(img_yuv, mu, Sigma, 'B'))

        # print(mu)
        # print('---' * 10)
        # print(Sigma)
        # print('---' * 10)
        # print(W_iF)
        # print('---' * 10)
        # print(W_iB)
        # print('---' * 10)
        # print(W_iF.shape)
        # print('---' * 10)
        # print(W_iB.shape)
        # print('---' * 10)
        # print(bot_n_topw_ij.shape)
        # print('---' * 10)
        # print(left_n_rightw_ij.shape)
        # print('---' * 10)
        # print(bot_n_topw_ij)
        # print('---' * 10)
        # print(left_n_rightw_ij)

        return scrib
