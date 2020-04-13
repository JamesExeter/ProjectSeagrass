import cv2
import os
import numpy as np
import sys

#Class primarily concerned with loading and resizing the image dataset during resizing and cropping operations
#Also used to convert the loaded data into csv files for the model to use.
class FileLoader:
    def __init__(self, path, save_path=None):
        self.load_from = path
        self.save_to = save_path
    def load(self, image, path_present):
        try:
            plot_img = self.create_image(image, path_present)
            return plot_img
        except FileNotFoundError:
            print("Could not open and load the image")
    def create_image(self, image, path_present):
        try:
            if (path_present):
                return cv2.imread(image)
            else:
                full_path = self.load_from + image
                return cv2.imread(full_path)
        except FileNotFoundError as fnf:
            print("File not found, error: " + str(fnf))
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        if height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = width, height
        return cv2.resize(image, dim, interpolation=inter)
    def save_image_to_file(self, image, image_name):
        if not self.save_to is None:
            try:
                cv2.imwrite(self.save_to + image_name, image)
            except IOError:
                print("There was an error saving to file")
                sys.exit()
        else:
            print("No path provided to save the image to")
    def load_first(self):
        for (root, _, filenames) in os.walk(os.path.normpath(self.load_from)):
            for file in filenames:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    first = os.path.join(root, file)
                    img = self.load(first, True)
                    if img is not None:
                        return np.array(img)
        
        return None

    def load_all_from_folder(self, output_name):
        list_of_files = []
        images = []
        names = []

        for (root, _, filenames) in os.walk(os.path.normpath(self.load_from)):
            for file in filenames:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".JPG")):
                    list_of_files.append(os.path.join(root, file))

        for filename in list_of_files:
            img = self.load(filename, True)
            if img is not None:
                images.append(img)
                names.append(filename)
                if output_name:
                    print("Loaded image: {}".format(filename))

        out = list(zip(images, names))
        return sorted(out, key=lambda x: x[1])
    
    def get_save_path(self):
        return self.save_to
    
    def get_load_path(self):
        return self.load_from