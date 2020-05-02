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
import utils.msg as msg
import time
from numpy import save
from numpy import load
import sys
import matplotlib.pyplot as plt

BATCH_SIZE = 20
VALID_SIZE = 0.1
#used for train, test, validation split of 70:20:10, split validiation from rest with 90:10, then split 90% into 80:20 using 90*(0.2222)
TEST_SIZE = 0.2222
total_prediction_time = 0
SHOW_IMAGES_WITH_PREDICTIONS = False

def shuffle_data(images, labels):
    return shuffle(images, labels)

def batch(data, batch_size):
    # For item i in a range that is a length of l,
    for i in range(0, len(data), batch_size):
        # Create an index range for l of n items:
        yield data[i:i+batch_size]

def train_in_batch(images, labels, cp_path, m_path, batch_size=BATCH_SIZE):
    images_batched = list(batch(images, batch_size))
    labels_batched = list(batch(labels, batch_size))
    
    height = rgb_images.shape[1]
    width = rgb_images.shape[2]
    depth = rgb_images.shape[3]
    print("\n")
    msg.timemsg("Training CNN start")

    model = cnn.create_cnn(width, height, depth)
    cnn.give_summary(model)
    
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
        
        #shuffles the data in batches
        #images_batched[i], labels_batched[i] = shuffle_data(images_batched[i], labels_batched[i])
        
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
            model = cnn.create_cnn(width, height, depth)
            msg.timemsg("Loading weights for model")
            model = cnn.load_weights_from_disk(model, cp_path)
        
        msg.timemsg("Batch {}: Training batch".format(i))
        model = cnn.train_model(model, train_images, train_labels, test_images, test_labels, 5, cp_path)
        
        msg.timemsg("Batch {}: Evaluating model".format(i))
        train_mse, test_mse, acc = cnn.evaluate_model(model, train_images, train_labels, test_images, test_labels)
        #can probably use train mse and test mse in plot training results method
        msg.timemsg("Batch {}: Model training MSE: {}, Model testing MSE: {}".format(i, train_mse, test_mse))
        msg.timemsg("Batch {}: Model accuracy: {:5.2f}%".format(i, 100*acc))
        
    msg.timemsg("Training CNN finished")
    msg.timemsg("Saving model to file")
    cnn.save_model(model, m_path)
    cnn.plot_model_accuracy(model)
    cnn.plot_model_loss(model)
    
    return model
    
def prediction(valid_imgs, sg_model, show_image=SHOW_IMAGES_WITH_PREDICTIONS):
    counter = 1
    n_valid = len(valid_imgs)
    results = []
    
    for img in valid_imgs:
        start_time = time.time()
        
        new_label = sg_model.predict(img)
        results.append(new_label)
        
        elapsed_time = (time.time() - start_time) * 1000 #ms
        global total_prediction_time
        total_prediction_time += elapsed_time
        
        msg.timemsg("Image Number: {}, Predicted label: {}".format(counter, new_label))
        msg.timemsg('Prediction Progress: {}%'.format(float(counter/n_valid)*100))
        counter += 1

        if show_image:
            # Display image
            plt.imshow(img, interpolation='nearest')
            # Add a title to the plot
            plt.title('Predicted: ' + str(new_label))
    
    return results
            
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

def scale_data(target_data):
    max_val = np.max(target_data)
    min_val = np.min(target_data)
    normalised = []

    for i in target_data:
        scaled = (i - min_val) / (max_val - min_val)
        normalised.append(scaled)

    return np.array(normalised) 

#method to allow the user to select a directory with images they want classified
def predict_directory():
    #load the entire model
    #get directory from user, maybe using file selector
    #process each image one at a time, load the image, normalise the data, make the prediction and then log the prediction
    pass

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
    
    if (args.skip_training == "1"):
        predict_directory()
    else:
        checkpoint_path = args.results_dir + args.checkpoint_dir
        model_path = args.results_dir + args.model_dir

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
        
        to_batch_images, valid_images, to_batch_labels, valid_labels = train_test_split(rgb_images, labels_arr, test_size=VALID_SIZE, random_state=1)
        
        cnn.ini()

        seagrass_model = train_in_batch(to_batch_images, to_batch_labels, checkpoint_path, model_path)
        
        msg.timemsg("Running predicitons on model using validation set")
         #convert the data to be in the range of 0 and 1 for the pixel values
        msg.timemsg("Normalising pixel values for validation set")
        valid_images = valid_images.astype('float32')
        valid_images /= 255.0
        msg.timemsg("Normalised prediction set pixel values")
        predictions = prediction(valid_images, seagrass_model, SHOW_IMAGES_WITH_PREDICTIONS)
        
        msg.timemsg("Predictions made on validation set, time taken: {}".format(total_prediction_time))
        
        #evaluate the predictions
        msg.timemsg("Evaluating the predictions made on the validation set")
        
        mse = mean_squared_error(valid_labels, predictions)
        msg.timemsg("Mean Squared Error on validation set: {}".format(mse))
        
        cnn.close()
    
    msg.timemsg("Execution finished, exiting")
