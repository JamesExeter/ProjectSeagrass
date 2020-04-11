#!/usr/bin/env python

import re
import numpy as np
import cv2
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import classifier.cnn
from utils.FileUtil import FileUtil
from utils.FileLoader import FileLoader
from sklearn.model_selection import train_test_split
import utils.msg as msg
import time
from numpy import save
from numpy import load
import sys


BATCH_SIZE = 500
TEST_SIZE = 0.2

def shuffle_data(images, labels):
    return shuffle(images, labels)

def batch(data, batch_size):
    # For item i in a range that is a length of l,
    for i in range(0, len(data), batch_size):
        # Create an index range for l of n items:
        yield data[i:i+batch_size]

def train_in_batch(images, labels, results_path, checkpoint_path, model_checkpoint, batch_size=BATCH_SIZE):
    images_batched = list(batch(images, batch_size))
    labels_batched = list(batch(labels, batch_size))
    print("\n")

    load_checkpoint = False
    for i in range(len(images_batched)):
        if i > 0:
            load_checkpoint = True
            
        msg.timemsg("Batch {}".format(i))
        
        #if a proper training set can't be made from the last batch, add the last batch to the one prior
        if len(images_batched[-1]) * TEST_SIZE < 1:
            last_images = images_batched[-1]
            last_labels = labels_batched[-1]
            del images_batched[-1]
            del labels_batched[-1]
            images_batched[-1] = np.append(images_batched[-1], last_images, axis=0)
            labels_batched[-1] = np.append(labels_batched[-1], last_labels, axis=0)

        #split data into training and testing data for that batch
        train_images, test_images, train_labels, test_labels = train_test_split(images_batched[i], labels_batched[i], test_size=TEST_SIZE, random_state=123)
        #input size for input layer is: 576x576 = 331776 neurons in input layer per image colour channel, 331776 * 3 per images
        
        # will need to train
        # checkpoints will be created during training
        # load from checkpoint if not the first batch and save the entire model at the end of each batch 
        
        
# calculate mean squared error
def mean_squared_error(actual, predicted):
    sum_square_error = 0.0
    for i in range(len(actual)):
        sum_square_error += (actual[i] - predicted[i])**2.0
    mean_square_error = 1.0 / len(actual) * sum_square_error
    return mean_square_error   

def convert_to_rbg(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_data(config_file):
    #split into batches of 500 for the computer to handle more easily
    start_time = time.time()
    msg.timemsg("Loading data from file")
    loader = FileUtil(config_file)
    loader.read_data()

    img_dir = loader.images[0].split("/")
    img_dir = args.root_img_dir + img_dir[1]

    img_loader = FileLoader(img_dir, args.root_img_dir)
    msg.timemsg("Loading images from folder")
    images = img_loader.load_all_from_folder(False)
    msg.timemsg("Images loaded from folder")

    msg.timemsg("Sorting image order and storing as numpy array")
    images.sort(key=lambda f: int(re.sub(r'\D', '', f[1])))
    images = np.array([(img[0]) for img in images])
    msg.timemsg("Images sorted and stored as numpy array")

    msg.timemsg("Converting images to RGB")
    for img in range(len(images)):
        images[img] = convert_to_rbg(images[img])
        
    msg.timemsg("Converted images to RGB")

    loaded_labels_arr = np.array([item for sublist in loader.labels for item in sublist])
    msg.timemsg("Loaded labels from file")

    return images, loaded_labels_arr

def scale_data(target_data):
    max_val = np.max(target_data)
    min_val = np.min(target_data)
    normalised = []

    for i in target_data:
        scaled = (i - min_val) / (max_val - min_val)
        normalised.append(scaled)

    return np.array(normalised) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #the name of the file that stores the images and corresponding coverages
    parser.add_argument("--graph", help="the root to the inception net folder")
    parser.add_argument("--root_img_dir", help="the root image directory of the project")
    parser.add_argument("--image_data_file", help="the name of the file containing the image paths and coverages")
    parser.add_argument("--results_dir", help="the name of the directory to store results to")
    parser.add_argument("--logging_file", help="name of the file to log to")
    parser.add_argument("--checkpoint_dir", help="the directory checkpoints are saved to during training")
    parser.add_argument("--model_dir", help="the directory models are saved to during training")
    args = parser.parse_args()

    msg.ini(args.results_dir + args.logging_file)

    checkpoint_path = args.results_dir + args.checkpoint_dir
    model_path = args.results_dir + args.model_dir

    rgb_images = np.array([])
    labels_arr = np.array([])

    #try to load the data, generate it if it doesnt load
    try:
        rgb_images = load(args.root_img_dir + "/images.npy")
        labels_arr = load(args.root_img_dir + "/labels.npy")
    except FileNotFoundError:
        msg.timemsg("Numpy files don't exist, generating them instead")
        msg.timemsg("Generating numpy data and saving to file")
        rgb_images, labels_arr = load_data(args.image_data_file)
        if len(rgb_images) > 0:
            save(args.root_img_dir + "images.npy", rgb_images)
            save(args.root_img_dir + "labels.npy", labels_arr)
            msg.timemsg("Data saved as numpy files, stored at: {}".format(args.root_img_dir))
        else:
            msg.timemsg("No data loaded, check for correct path and file name parameters")
            sys.exit()

    msg.timemsg("Loaded data from numpy files")

    height = rgb_images.shape[1]
    width = rgb_images.shape[2]
    depth = rgb_images.shape[3]

    # fit and transform in one step for the labels
    labels_arr = scale_data(labels_arr)

    #at this point I have loaded in all of the images in RGB form into a numpy array in order
    #the labels are also loaded in properly too as floats
    #the data can now be shuffled whilst maintaining the relationship between labels and images
    #and then broken up into training and testing data to be fed into the model
    #the input shape for the input layer will be 576X576

    #shuffles the data before batching
    rgb_images, labels_arr = shuffle_data(rgb_images, labels_arr)

    train_in_batch(rgb_images, labels_arr, args.results_dir, checkpoint_path, model_path)
    
    #either have a model returned or load it from file if it exists already
    #then run predictions on it for testing
    #or evaluate the model and predict data needed