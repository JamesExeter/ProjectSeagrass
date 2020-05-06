#!/usr/bin/env python

import re
import numpy as np
import cv2
import argparse
from sklearn.utils import shuffle
import classifier.cnn as cnn
from utils.FileUtil import FileUtil
from utils.FileLoader import FileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import utils.msg as msg
import time
from numpy import save
from numpy import load
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os

"""
Class used to train the classifier or make predictions with if the classifier is trained
Needs to be run using either a bash script or with all of the variables required by the args parser in the main method
"""

#variables needed to process the dataset for training
BATCH_SIZE = 10
VALID_SIZE = 0.1
#used for train, test, validation split of 70:20:10, split validiation from rest with 90:10, then split 90% into 80:20 using 90*(0.2222)
TEST_SIZE = 0.2222
total_prediction_time = 0
SHOW_IMAGES_WITH_PREDICTIONS = False

#shuffles the images and labels, keeping each pair together, made obsolete due to the sklearn module
def shuffle_data(images, labels):
    return shuffle(images, labels)

#batches the data based on a batch size provided
def batch(data, batch_size):
    #case where less data than batch size, if this is the case, batching is not needed
    if (len(data) < batch_size):
        yield data
    else:
        # For item i in a range that is a length of l,
        for i in range(0, len(data), batch_size):
            # Create an index range for l of n items:
            yield data[i:i+batch_size]

#traines the neural network in batches, creating manual batches rather than using
#pre-built modules to help control memory management
def train_in_batch(images, labels, cp_path, m_path, batch_size=BATCH_SIZE):
    #batches the labels and images
    images_batched = list(batch(images, batch_size))
    labels_batched = list(batch(labels, batch_size))
    
    #all image shapes are identical so choose the dimensions of the first
    height = rgb_images.shape[1]
    width = rgb_images.shape[2]
    depth = rgb_images.shape[3]
    print("\n")
    msg.timemsg("Training CNN start")

    #create the model
    model = cnn.create_cnn(width, height, depth)
    cnn.give_summary(model)
    
    #train each batch one at a time, with evaluation performed
    for i in range(len(images_batched)):        
        msg.timemsg("Batch {}".format(i))
        
        #if a proper training set can't be made from the last batch, add the last batch to the one prior
        if len(images_batched[-1]) * TEST_SIZE < 1:
            last_images = images_batched[-1]
            last_labels = labels_batched[-1]
            del images_batched[-1]
            del labels_batched[-1]
            images_batched[-1] = np.append(images_batched[-1], last_images, axis=0)
            labels_batched[-1] = np.append(labels_batched[-1], last_labels, axis=0)

        #convert the data to be in the range of 0 and 1 for the pixel values
        msg.timemsg("Batch {}: Normalising pixel values".format(i))
        images_batched[i] = images_batched[i].astype('float32')
        images_batched[i] /= 255.0
        msg.timemsg("Batch {}: Normalised pixel values".format(i))
        
        #split data into training and testing data for that batch
        #also shuffles the data whilst splitting
        msg.timemsg("Batch {}: Shuffling and splitting data for training".format(i))
        train_images, test_images, train_labels, test_labels = train_test_split(images_batched[i], labels_batched[i], test_size=TEST_SIZE, random_state=42)

        msg.timemsg("Batch {}: Data shuffled and split, it is ready for usage".format(i))
        #input size for input layer is: 576x576 = 331776 neurons in input layer per image colour channel, 331776 * 3 per images
        
        # will need to train
        # checkpoints will be created during training
        # load from checkpoint if not the first batch and save the entire model at the end of each batch
        
        if i > 0:
            #if not the first batch, then load the weights from the previous batch and begin training again
            model = cnn.create_cnn(width, height, depth)
            msg.timemsg("Loading weights for model")
            model = cnn.load_weights_from_disk(model, cp_path)
        
        msg.timemsg("Batch {}: Training batch".format(i))
        model = cnn.train_model(model, train_images, train_labels, test_images, test_labels, 5, cp_path)
        
        msg.timemsg("Batch {}: Evaluating model".format(i))
        m_s_error, mean_abs_error, = cnn.evaluate_model(model, test_images, test_labels)
        #can probably use train mse and test mse in plot training results method
        msg.timemsg("Batch {}: test loss: {:.5f}, test mae: {:.5f}\n\n".format(i, m_s_error, mean_abs_error))
        
        #model = cnn.create_cnn(width, height, depth)
        #msg.timemsg("Loading weights for testing model weight loading")
        #model = cnn.load_weights_from_disk(model, cp_path)
        #msg.timemsg("Batch {}: Evaluating model second time".format(i))
        #m_s_error, mean_abs_error, = cnn.evaluate_model(model, test_images, test_labels)
        #can probably use train mse and test mse in plot training results method
        #msg.timemsg("Batch {}: test loss: {:.5f}, test mae: {:.5f}\n\n".format(i, m_s_error, mean_abs_error))
        
    msg.timemsg("Training CNN finished")
    msg.timemsg("Saving model to file")
    cnn.save_model(model, m_path)
    
    #plot mse and mae of final trained model
    plotter = cnn.create_history_plotter()
    cnn.plot_mse(model, plotter)
    cnn.plot_mae(model, plotter)
    
    return model

#given a trained model and validation images, the model makes predictions
#show_image variables is a boolean that if true, shows the image with the prediction
def prediction(valid_imgs, sg_model, show_image=SHOW_IMAGES_WITH_PREDICTIONS):
    counter = 1
    n_valid = len(valid_imgs)
    results = np.array([])
    
    #go through each image one at a time and make a prediction, adding the prediction to the output array
    for img in valid_imgs:
        start_time = time.time()
        
        new_label = sg_model.predict(np.array([img]))
        results = np.append(results, new_label, axis=None)
        
        elapsed_time = (time.time() - start_time) * 1000 #ms
        global total_prediction_time
        total_prediction_time += elapsed_time
        
        msg.timemsg("Image #: {}, Predicted label: {:.4f}, Predicted in: {:.3f}s".format(counter, float(new_label), elapsed_time / 1000.0))
        msg.timemsg('Prediction Progress: {:.2f}%'.format(float(counter/n_valid)*100))
        counter += 1

        if show_image:
            # Display image
            plt.imshow(img, interpolation='nearest')
            # Add a title to the plot
            plt.title('Predicted: ' + str(new_label))
    
    return results

#converts loaded images to rgb as default is bgr
def convert_to_rbg(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#given a configuration file
#it loads all of the images and labels in that file in order for numpy files to be made for the images and labels
def load_data(config_file):
    #split into batches of 50 for the computer to handle more easily
    msg.timemsg("Loading data from file")
    loader = FileUtil(config_file)
    loader.read_data()

    img_dir = loader.images[0].split("/")
    img_dir = args.root_img_dir + "/" + img_dir[1]

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

#scales the data of a given array to be between 0 and 1, can be used for images and labels,
#but if memory is an issue, just divided each pixel by 255 in place to save memory
def scale_data(target_data):
    max_val = np.max(target_data)
    min_val = np.min(target_data)
    normalised = []

    for i in target_data:
        scaled = (i - min_val) / (max_val - min_val)
        normalised.append(scaled)

    return np.array(normalised) 

#method to allow the user to select a directory with images they want classified
def predict_directory(model_to_load, results):
    counter = 0
    start_time = time.time()
    #initialise the cnn instance so we have access to it's functionality
    cnn.ini()
    
    #load the entire model
    model = None
    try:
        model = cnn.load_model_from_disk(model_to_load)
    except ValueError:
        msg.timemsg("No model loaded, check the path is correct or a model has been saved properly")
        
    if (model is not None):
        #get directory from user, maybe using file selector
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title='Select Folder')
        
        out_file_name = os.path.join(results, "Predictions.txt")
        
        with open(out_file_name, "w") as prediction_file:
            #process each image one at a time
            #Check first with the CheckValidImages class that each file is an image
            #This can be done by running the class in the command line and passing the path of the directory required
            for image in os.scandir(path):
                if image.is_file():
                    name = image.name
                    
                    msg.timemsg("Loading image: {}".format(name))
                    
                    image_path = os.path.join(path, image.name)
                    numpy_image = cv2.imread(image_path)
                    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
                    numpy_image = numpy_image.astype('float32')
                    numpy_image /= 255.0
                    numpy_image = np.array([numpy_image])
                    
                    msg.timemsg("Making prediction on: {}".format(name))
                    predicted_coverage = model.predict(numpy_image)
                    
                    elapsed_time = (time.time() - start_time) * 1000 #ms
                    global total_prediction_time
                    total_prediction_time += elapsed_time
                    #load the image, normalise the data, make the prediction and then log the prediction
                    print("Image: {}, Prediction: {:.3f}, Predicted in: {:.3f}s".format(name, float(predicted_coverage), elapsed_time / 1000.0), file=prediction_file)
                    
                    counter += 1
        
        msg.timemsg("All predictions made and stored at: {}".format(out_file_name))
        
    return counter
    
#main argument that loads all of the arguments from a bash script or command line
#either trains a model using the given data
#or loads a trained model to make predictions with
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
    parser.add_argument("--using_small", help="determines whether the small or large dataset is being used")
    parser.add_argument("--skip_training", help="determines whether to skip training and load a pre-trained model instead")
    args = parser.parse_args()

    msg.ini(args.results_dir + args.logging_file)
    
    checkpoint_path = args.results_dir + args.checkpoint_dir
    model_path = args.results_dir + args.model_dir
    #if training not needed, make predictions on formatted images in a given directory
    if (args.skip_training == "1"):
        number_images = predict_directory(model_path, args.results_dir)
        if number_images > 0:
            msg.timemsg("Predictions made on set of {} images, time taken: {:.3f}s".format(number_images, float(total_prediction_time / 1000.0)))
            msg.timemsg("Average prediction time of {:.3f}s per image".format(float((total_prediction_time / 1000.0) / number_images)))
    else:
        #train the model and generate the evaluation metrics
        rgb_images = np.array([])
        labels_arr = np.array([])
        
        msg.timemsg("Loading or creating numpy data")
        
        IMAGE_FILE = "/images.npy"
        LABEL_FILE = "/labels.npy"
        if(args.using_small == "1"):
            IMAGE_FILE = "/images_small.npy"
            LABEL_FILE = "/labels_small.npy"

        #try to load the data, generate it if it doesnt load
        try:
            rgb_images = load(args.root_img_dir + IMAGE_FILE)
            labels_arr = load(args.root_img_dir + LABEL_FILE)
        except FileNotFoundError:
            msg.timemsg("Numpy files don't exist, generating them instead")
            msg.timemsg("Generating numpy data and saving to file")
            
            rgb_images, labels_arr = load_data(args.image_data_file)
            if len(rgb_images) > 0:
                save(args.root_img_dir + IMAGE_FILE, rgb_images)
                save(args.root_img_dir + LABEL_FILE, labels_arr)
                msg.timemsg("Data saved as numpy files, stored at: {}".format(args.root_img_dir))
                msg.timemsg("Restart the program so the data can be reloaded for proper memory management")
                sys.exit(0)
            else:
                msg.timemsg("No data loaded, check for correct path and file name parameters")
                sys.exit()

        msg.timemsg("Loaded data from numpy files")

        # fit and transform in one step for the labels
        
        msg.timemsg("Normalising labels")
        labels_arr = scale_data(labels_arr)
        msg.timemsg("Labels normalised")

        #at this point I have loaded in all of the images in RGB form into a numpy array in order
        #the labels are also loaded in properly too as floats
        #the data can now be shuffled whilst maintaining the relationship between labels and images
        #and then broken up into training and testing data to be fed into the model
        #the input shape for the input layer will be 576X576
        
        #generate validation data, and data for training / testing
        to_batch_images, valid_images, to_batch_labels, valid_labels = train_test_split(rgb_images, labels_arr, test_size=VALID_SIZE, random_state=1)
        
        cnn.ini()

        #generate a model by training it
        seagrass_model = train_in_batch(to_batch_images, to_batch_labels, checkpoint_path, model_path)
        
        msg.timemsg("Running predicitons on model using validation set")
        #convert the data to be in the range of 0 and 1 for the pixel values
        msg.timemsg("Normalising pixel values for validation set")
        valid_images = valid_images.astype('float32')
        #would use the scale data method here but it uses more memory, and we know the upper
        #limit value of rgb pixels is always the same
        #this is more efficient for that task and is in place
        valid_images /= 255.0
        msg.timemsg("Normalised prediction set pixel values")
        
        predictions = prediction(valid_images, seagrass_model, SHOW_IMAGES_WITH_PREDICTIONS)
        msg.timemsg("All predictions made on validation set of {} images, time taken: {:.3f}s".format(len(valid_images), float(total_prediction_time / 1000.0)))
        msg.timemsg("Average prediction time of {:.3f}s per image\n".format(float((total_prediction_time / 1000.0) / len(valid_images))))
        
        #evaluate the predictions
        msg.timemsg("Evaluating the predictions made on the validation set")
        
        #error measurements of the predictions  
        mse = mean_squared_error(valid_labels, predictions)
        mae = mean_absolute_error(valid_labels, predictions)
        
        msg.timemsg("Mean Squared Error on validation set: {:.5f}".format(mse))
        msg.timemsg("Mean Absolute Error on validation set: {:.5f}\n".format(mae))
        
        cnn.plot_predictions_vs_actual(valid_labels, predictions)
        cnn.plot_prediction_error_distribution(valid_labels, predictions)
        
        cnn.close()
    
    msg.timemsg("Execution finished, exiting")
