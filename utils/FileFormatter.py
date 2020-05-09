"""
image manipulation for testing seagrass images and features
"""

import os, os.path
import time
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import random as rand
import itertools
from utils.ImageVisualiser import ImageVisualiser
from utils.ImageEditor import ImageEditor
from utils.ImageEditor import find_cropping_coordinates
from utils.ImageEditor import crop_quadrat
from utils.FileLoader import FileLoader
import utils.msg as msg
import argparse

"""
Class primarily used to run all of the images and label formatting by bringing them together
and individually performing different formatting procedures based on different used methods
Also includes some extra functions to pre-emptively format images
"""

#squares an image prior to any further formatting
def initial_square(image, border_vals):
    height = image.shape[0]
    width = image.shape[1]

    middle_y = (border_vals[0] + border_vals[1]) // 2
    middle_x = (border_vals[2] + border_vals[3]) // 2

    left, right, top, bottom = 0, width, 0, height

    if(height > width):
        top = middle_y - middle_x
        bottom = middle_y + middle_x
    else:
        left = middle_x - middle_y
        right = middle_x + middle_y

    return crop_quadrat(image, top, bottom, left, right)

#finds the smallest image in a set of given image dimensions, ensuring that
#it is close to being square too
def find_smallest_dimensions(dimensions):
    smallest_width = sorted(dimensions)[0][1]
    smallest_aspect_height = sorted(dimensions)[0][0]

    avg_aspect = calculate_average_aspect_ratio(dimensions)
    smallest_ratio = smallest_width / smallest_aspect_height

    if (avg_aspect <= smallest_ratio):
        return (smallest_width, int(smallest_width/avg_aspect))

    return (smallest_width, smallest_aspect_height)

#caculates the average aspect ratio of a dataset of images
def calculate_average_aspect_ratio(dimensions):
    res = sum(i[0] for i in dimensions), sum(i[1] for i in dimensions)
    return (res[1] / res[0]) 

#renames all of the images, this is used for either a fresh dataset or for adding new images
#to an existing dataset whilst ensuring the naming increments properly
def process_for_saving(saver, images):
    path = saver.get_save_path()
    num_current_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    for i in range(num_current_files, (num_current_files + len(images))):
        #FSI stands for Formatted Seagrass Image
        name = "FSI" + str(i) + ".jpg"
        saver.save_image_to_file(images[i-num_current_files], name)

#renames all of the images, starting from a given number
def process_for_saving_just_rename(saver, images, min_num):
    path = saver.get_save_path()
    count = min_num
    for img in images:
        #FSI stands for Formatted Seagrass Image
        name = "FSI" + str(count) + ".jpg"
        saver.save_image_to_file(img, name)
        count += 1

#call this to save and process the images, no need to extract the data from the images prior
def resize_and_save(saver, images, min_width=576, min_height=576):
    #used to find the smallest image in a dataset
    dimensions = [(img.shape[0], img.shape[1]) for img in images]
    new_min_width, new_min_height = find_smallest_dimensions(dimensions)

    #if adding to a new dataset, if an image in the already saved data
    #is smaller than the smallest in the new dataset, then switch to the new smaller size
    if (min_width is None or min_height is None):
        min_width = new_min_width
        min_height = new_min_height
    else:
        #the new smallest size is the overall smallest
        if(new_min_width < min_width):
            min_width = new_min_width
        if(new_min_height < min_height):
            min_height = new_min_height
            
    #saves all of the images after resizing
    resized = []
    for img in images:
        resized.append(saver.resize_with_aspect_ratio(img, min_width, min_height))

    process_for_saving(saver, resized)

#used to rename the images in a dataset using the required format
def renaming_main(img_dir):
    start_time = time.time()
    msg.ini("/home/james/Documents/Seagrass-Repository/Results/output_log.txt")

    rename_loader = FileLoader(img_dir, img_dir)
    to_rename = rename_loader.load_all_from_folder(False)
    to_rename.sort(key=lambda f: int(re.sub('\D', '', f[1])))
    to_rename = np.array([(img[0]) for img in to_rename])

    msg.timemsg("Loaded {} images to be renamed".format(len(to_rename)))
    process_for_saving_just_rename(rename_loader, to_rename, 145)

#methods used for testing feature extraction
def testing_feature_extraction_main():
    img_path = "/home/james/Documents/Seagrass-Repository/ProjectSeagrass/"
    save_path = "/home/james/Documents/Seagrass-Repository/ProjectSeagrass/"
    test_image = "Test.JPG"

    loader = FileLoader(img_path, save_path)
    image = loader.load(test_image, False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    edged_image = ImageEditor(image).view_edges()
    sobel_x_image = ImageEditor(image).view_sobel_x()
    sobel_y_image = ImageEditor(image).view_sobel_y()
    laplacian_image = ImageEditor(image).view_laplacian()

    #thresholds an image to be black and white
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #displays  series of images showing various transformations to them
    image_data = [['Original', image], ['Canny', edged_image], ["Sobel X", sobel_x_image],
    ["Sobel Y", sobel_y_image], ["Laplacian", laplacian_image], ["Thresholded", thresh]]
    ImageVisualiser.display_image(image_data, 3)

    #Testing the usefullness of the grabcut algorithm
    #Grabcut is not wholly effective 
    #ImageEditor(image).grab_cut_seagrass()

    #Visualising the colour ranges in an image to see which colour to base processing on
    #Will hurt PC
    #Trying to display 7257411 pixels across 3 colour channels in a 3D space
    #ImageVisualiser(image).visualise_rgb()
    #ImageVisualiser(image).visualise_hsv()

    dark_green = (50, 90, 80)
    light_green = (200, 210, 170)
    ImageEditor(image).color_segment(dark_green, light_green)

    #Cropping images based on their quadrats
    white_mask = ImageEditor(image).generate_quadrat_mask()
    #silver_mask = ImageEditor(image).generate_silver_mask()
    cropped_image = ImageEditor(image).crop_quadrat_from_image(white_mask)
    
    ImageVisualiser.display_image([["Original", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)], 
    ["Cropped", cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)]], 2)

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    #saves an image to a folder, ensuring rgb properties preserved
    process_for_saving(loader, [cropped_image])

#methods used to augment a dataset, generating an augmentation 6 times per image
def augmentation_main():
    save_path = "/home/james/Documents/Seagrass-Repository/Images/Formatted_Images/"
    load_path = "/home/james/Documents/Seagrass-Repository/Images/Formatted_Images/"

    start_time = time.time()
    msg.ini("/home/james/Documents/Seagrass-Repository/Results/output_log.txt")

    aug_loader = FileLoader(load_path, save_path)
    to_augment = aug_loader.load_all_from_folder(False)
    to_augment.sort(key=lambda f: int(re.sub('\D', '', f[1])))
    to_augment = np.array([(img[0]) for img in to_augment])

    msg.timemsg("Loaded {} images to be augmented".format(len(to_augment)))
    
    loader2 = FileLoader(save_path, save_path)
    check_saved = loader2.load_first()
    
    augmented_images = []
    sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    blurring_kernel = np.ones((5,5), np.float32) / 25

    # possible data augmentations for each image:
    # a vertical or horizontal flip, or none
    # a change in brightness, contrast or blurring
    # decide which image alteration to perform: sharpen, blur, adjust contrast or none
    # then for that alteration, select the image orientation, vert, hori or none
    # repeat this 4 times until each image alteration  

    # each number refers to an operation function to be applied to the image
    operations = [1,2,3,4]
    rand.shuffle(operations)
    # each number refers to which orientation to use as described above
    orientation = [1,2,3]
    rand.shuffle(orientation)

    all_combinations = list(itertools.product(operations, orientation))
    all_combinations.remove((4,3))
    rand.shuffle(all_combinations)

    msg.timemsg("Beginning augmentation of {} images".format(len(to_augment)))

    if len(to_augment) > 0:
        min_height = None
        min_width = None
        
        if(check_saved is not None):
            min_width = check_saved.shape[1]
            min_height = check_saved.shape[0]

        msg.timemsg("Width and Height Parameters set with Height: {} and Width {}".format(min_height, min_width))
        to_perform = all_combinations[0:6]

        for combination in to_perform:
            for i in range(len(to_augment)):
                new_image = to_augment[i]
                editor = ImageEditor(new_image)
                #perform operation picked from random choice
                if combination[0] == 1:
                    g_val = rand.uniform(0.8, 2)
                    new_image = editor.alter_brightness(g_val)        
                if combination[0] == 2:
                    new_image = editor.alter_sharpness(sharpening_kernel)
                if combination[0] == 3:
                    new_image = editor.blur_image(blurring_kernel)
                
                #flip image if chosen to be
                if combination[1] == 1:
                    new_image = editor.flip_image_vert()
                if combination[1] == 2:
                    new_image = editor.flip_image_hor()

                augmented_images.append(np.array(new_image))

        msg.timemsg("Augmentation done, resizing and saving the images")
        resize_and_save(aug_loader, augmented_images, min_width, min_height)
    else:
        print("Directory empty or problem loading images")

    msg.timemsg("Finished executing")

#resizes all images in the dataset to the required size, change the directory save and load path to the one needed
def resizing_main():
    save_path = "/home/james/Documents/Seagrass-Repository/Images/Emma_Images/"
    load_path = "/home/james/Documents/Seagrass-Repository/Images/Emma_Images/"
    width = 576
    height = 576

    start = time.time()
    msg.ini("/home/james/Documents/Seagrass-Repository/Results/output_log.txt")

    loader = FileLoader(load_path, save_path)
    to_process = loader.load_all_from_folder(False)
    to_process.sort(key=lambda f: int(re.sub('\D', '', f[1])))

    if(len(to_process) > 0):
        msg.timemsg("Width and Height Parameters set with Height: {} and Width {}".format(height, width))
        images = np.array([(img[0]) for img in to_process])

        msg.timemsg("Processing done, resizing and saving")
        resize_and_save(loader, images, width, height)
    else:
        print("Directory empty or loading problem occurred")

    msg.timemsg("Finished executing")

#experiments all key features of the formatting process works properly
def main(load_path, save_path):    
    start = time.time()
    msg.ini("/home/james/Documents/Seagrass-Repository/Results/output_log.txt")

    loader = FileLoader(load_path, save_path)
    to_process = loader.load_all_from_folder(False)

    loader2 = FileLoader(save_path, save_path)
    check_saved = loader2.load_first()

    if(to_process != []):
        min_height = 576
        min_width = 576
        
        if(check_saved != []):
            min_width = check_saved.shape[1]
            min_height = check_saved.shape[0]

        msg.timemsg("Width and Height Parameters set with Height: {} and Width {}".format(min_height, min_width))
        images = np.array([(img[0]) for img in to_process])
        out = []
        for i in range(len(images)):
            #need to adjust the quadrat cropping slightly
            initial_mask = ImageEditor(images[i]).generate_silver_mask()
            border_vals = find_cropping_coordinates(initial_mask, True)
            squared = initial_square(images[i], border_vals)
            cropped_mask = ImageEditor(squared).generate_silver_mask()
            out.append(ImageEditor(squared).crop_quadrat_from_image(cropped_mask))

        msg.timemsg("Processing done, resizing and saving")
        resize_and_save(loader, out, min_width, min_height)
    else:
        print("Directory empty or loading problem occurred")

    msg.timemsg("Finished executing")

#used to run different main methods    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_folder", help="the folder pre-formatting images are loaded from")
    parser.add_argument("--save_folder", help="the folder the formatted images are saved too")
    args = parser.parse_args()
    
    #main(args.load_folder, args.save_folder)
    #testing_feature_extraction_main()
    #augmentation_main()
    #renaming_main()
    #resizing_main()
