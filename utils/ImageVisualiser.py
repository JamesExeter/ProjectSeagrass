import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np

class ImageVisualiser:
    def __init__(self, image_vis):
        self.image_vis = image_vis
        self.image_vis_hsv = self.convert_image()
    def convert_image(self):
        return cv2.cvtColor(self.image_vis, cv2.COLOR_BGR2HSV)
    def visualise_rgb(self):
        r, g, b = cv2.split(self.image_vis)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_cols = self.image_vis.reshape((np.shape(self.image_vis)[0]*np.shape(self.image_vis)[1], 3))
        norm = colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_cols)
        pixel_cols = norm(pixel_cols).tolist()

        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_cols, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        plt.show()
    def visualise_hsv(self):
        h, s, v = cv2.split(self.image_vis_hsv)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_cols = self.image_vis_hsv.reshape((np.shape(self.image_vis_hsv)[0]*np.shape(self.image_vis_hsv)[1], 3))
        norm = colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_cols)
        pixel_cols = norm(pixel_cols).tolist()

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_cols, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()

    @staticmethod
    def display_image(images, cols):
        number = len(images)
        rows = int((number /cols))
        if not (number % 3 == 0):
            rows += 1

        i = 1
        for title, image in images:
            plt.subplot(rows, 3, i), plt.imshow(image, cmap = 'gray')
            plt.title(title), plt.xticks([]), plt.yticks([])
            i += 1

        plt.show()