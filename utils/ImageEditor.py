import cv2
from matplotlib import pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

def find_min(vals, current_vals):
    if len(vals) > 0:
        return min(vals, key= lambda t: t[1])
    else:
        return current_vals

def find_max(vals, current_vals):
    if len(vals) > 0:
        return max(vals, key= lambda t: t[1])
    else:
        return current_vals

def crop_quadrat(image, min_y, max_y, min_x, max_x):
    return image[min_y:max_y, min_x:max_x]

def ensure_square(left, right, top, bottom, w, h):
    y_seperation = bottom[0] - top[0]
    x_seperation = right[1] - left[1]

    dif = max(x_seperation, y_seperation) - min(y_seperation, x_seperation)
    output = [left, right, top, bottom]

    #10% threshold difference
    if (dif >= ((w // 10) and (h // 10))):
        if (y_seperation > x_seperation):
            sep_min = ((w // 2) - (y_seperation // 2))
            sep_max = ((w // 2) + (y_seperation // 2))
            
            if (sep_min < 0):
                sep_min = ((w // 2) - (x_seperation // 2))
            if (sep_max > h):
                sep_max = ((w // 2) + (x_seperation // 2))

            output[0] = (left[0], sep_min)
            output[1] = (left[0], sep_max)
        else:
            sep_min = ((h // 2) - (x_seperation // 2))
            sep_max = ((h // 2) + (x_seperation // 2))
            if (sep_min < 0):
                sep_min = ((h // 2) - (y_seperation // 2))
            if (sep_max > w):
                sep_max = ((h // 2) - (y_seperation // 2))
            output[2] = (sep_min, top[1])
            output[3] = (sep_max, top[1])

    return output

def find_cropping_coordinates(mask, val_to_find):
    height = mask.shape[0]-1
    width = mask.shape[1]-1
 
    middle_x = width // 2
    middle_y = height // 2

    border_left, border_right, border_top, border_bottom = (middle_y, 0), (middle_y, width), (0, middle_x), (height, middle_x)

    x_change = width // 5
    y_change = height // 8

    max_x_range = width // 8
    max_y_range = height // 8

    x_constants = [middle_x, middle_x+x_change, middle_x-x_change, middle_x+2*x_change, middle_x-2*x_change]
    y_constants = [middle_y, middle_y+y_change, middle_y-y_change, middle_y+2*y_change, middle_y-2*y_change]

    #Border left
    vals = []
    for const_val in y_constants:
        found_quad = False
        
        for x in range(max_x_range):
            if((mask[const_val][x] == val_to_find) and not found_quad):
                vals.append((const_val, x))
                found_quad = True
    
    border_left = find_max(vals, border_left)
    
    #Border right
    vals = []
    for const_val in y_constants:
        found_quad = False

        for x in range(max_x_range):
            if((mask[const_val][width - x] == val_to_find) and not found_quad):
                vals.append((const_val, (width - x)))
                found_quad = True
            
    border_right = find_min(vals, border_right)
    
    #Border-top
    vals = []
    for const_val in x_constants:
        found_quad = False

        for y in range(max_y_range):
            if((mask[y][const_val] == val_to_find) and not found_quad):
                vals.append((y, const_val))
                found_quad = True
            
    border_top = find_max(vals, border_top)

    #Border-bottom
    vals = []
    for const_val in x_constants:
        found_quad = False

        for y in range(max_y_range):
            if((mask[height - y][const_val] == val_to_find) and not found_quad):
                vals.append((height - y, const_val))
                found_quad = True

    border_bottom = find_max(vals, border_bottom)

    border_left, border_right, border_top, border_bottom = ensure_square(border_left, border_right, 
    border_top, border_bottom, width, height)

    #perform a test to compare the size of the sides, if one side is larger than the other, balance the other side
    #print(border_top, border_bottom, border_left, border_right)
    return border_top[0], border_bottom[0], border_left[1], border_right[1]

class ImageEditor:
    def __init__(self, image_edit):
        self.image_edit = image_edit
        self.height, self.width, self.channels = self.image_edit.shape
    def access_pixels(self):
        return self.width, self.height, self.channels
    def view_edges(self):
        return cv2.Canny(self.image_edit, 100, 100)
    def view_sobel_x(self):
        return cv2.Sobel(self.image_edit, cv2.CV_64F, 1, 0, ksize=5)
    def view_sobel_y(self):
        return cv2.Sobel(self.image_edit, cv2.CV_64F, 0, 1, ksize=5)
    def view_laplacian(self):
        return cv2.Laplacian(self.image_edit, cv2.CV_64F)
    def grab_cut_seagrass(self):
        mask = np.zeros(self.image_edit.shape[:2], np.uint8)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (80, 10, 410, 380)

        cv2.grabCut(self.image_edit, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        self.image_edit = self.image_edit*mask2[:, :, np.newaxis]
        plt.imshow(self.image_edit)
        plt.colorbar()
        plt.show()
    def color_segment(self, lrc, urc):
        mask = cv2.inRange(self.image_edit, lrc, urc)
        result = cv2.bitwise_and(self.image_edit, self.image_edit, mask=mask)
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.show()
    def generate_quadrat_mask(self):
        gray = cv2.cvtColor(self.image_edit, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (17, 17), 0)
        
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        return thresh

    def crop_quadrat_from_image(self, mask, find_coodinates=False):
        #crops to the outside of the quadrat
        cropped_values_1 = find_cropping_coordinates(mask, 255)
        original_cropped = crop_quadrat(self.image_edit, cropped_values_1[0], cropped_values_1[1], cropped_values_1[2], cropped_values_1[3])
        outside_crop = crop_quadrat(mask, cropped_values_1[0], cropped_values_1[1], cropped_values_1[2], cropped_values_1[3])

        #crops to the inside of the quadrat
        cropped_values_2 = find_cropping_coordinates(outside_crop, 0)
        inside_crop = crop_quadrat(original_cropped, cropped_values_2[0], cropped_values_2[1], cropped_values_2[2], cropped_values_2[3])
        if find_coodinates:
            return cropped_values_2
        else:
            return inside_crop

    def generate_silver_mask(self):
        gray = cv2.cvtColor(self.image_edit, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        thresh = cv2.threshold(blur,135,255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return close

    def flip_image_hor(self):
        flip_hr = iaa.Fliplr(p=1.0)
        flip_hr_image = flip_hr.augment_image(self.image_edit)
        return flip_hr_image

    def flip_image_vert(self):
        flip_vrt = iaa.Fliplr(p=1.0)
        flip_vrt_image = flip_vrt.augment_image(self.image_edit)
        return flip_vrt_image

    def alter_brightness(self, gamma_val):
        contrast = iaa.GammaContrast(gamma=gamma_val)
        contrast_image = contrast.augment_image(self.image_edit)
        return contrast_image

    def alter_sharpness(self, kernel):
        image_sharp = cv2.filter2D(self.image_edit, -1, kernel)
        return image_sharp

    def blur_image(self, kernel):
        blurred = cv2.filter2D(self.image_edit, -1, kernel)
        return blurred