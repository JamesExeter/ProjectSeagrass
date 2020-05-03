import re
import os, os.path
import fnmatch
import argparse
import sys

"""
Class primarily foccussed on loading images and labels from a folder to be used in data augmentation or for training
Updates labels in the config file to accomodate the new labels
"""
class FileUtil:
    #takes a config file parameter to load the data from
    #ensures directory exists and initialises storage structures
    def __init__(self, path):
        self.name = None
        if self.check_exists(path):
            self.name = path 

        self.images = []
        self.labels = []

    #error handling if directory doesn't exist
    def check_exists(self, file):
        try:
            with open(file, "r") as file:
                return True
        except FileNotFoundError:
            return False
        except IOError:
            return False

    #reads the data from the config file if it exists
    def read_data(self):
        if self.name != None:
            with open(self.name, "r") as file:
                data = file.read().splitlines()
                for line in data:
                    #parse each line of the file one at a time
                    self.parse_line(line)
            
            file.close()
        else:
            print("No file loaded, please check file name and path")
            sys.exit()

    #used to parse each line individually
    def parse_line(self, line):
        #splits the config line into the image and label components
        l = line.split('"')[1::2]
        
        if(l != ""):
            #appends each image path and label to the respective lists 
            self.images.append(l[0])
            labels_buf = l[1]
        
            labels_buf = re.findall(r"[-+]?\d*\.\d+|\d+", labels_buf)
            self.labels.append([float(i) for i in labels_buf])

    #used to update the config file to add the augmented data
    def append_data(self, fname, flabel):
        if self.name != None:
            with open(self.name, "a+") as file:
                #can change Images/Formatted_Images to be dynamic with a variable
                to_write = '"Images/Formatted_Images/{}, "coverage: {}"\n'.format(fname, flabel)
                file.write(to_write)
        else:
            print("No file to write to, please check file name and path")

    #calculates the number of labels and entries that need to be added and creates the entries
    def update_labels(self, path_to_images):
        last_name = self.images[-1]
        temp = re.findall(r'\d+', last_name) 
        num_original = (list(map(int, temp))[0] + 1) # adding 1 since counting starts at 0
        
        #gets the number of augmented images that the labels needs to match
        max_files = len(fnmatch.filter(os.listdir(path_to_images), '*.jpg'))

        #gets the number of times the current labels need to be duplicated by
        n_repeats = int((max_files - num_original) / num_original)
        
        repeated_coverage = []
        for i in range(n_repeats):
            repeated_coverage.extend(self.labels)
        
        filenames = list(range(num_original, max_files))
        filenames = ['FSI' + str(s) + '.jpg"' for s in filenames]

        for i in range(len(filenames)):
            self.append_data(filenames[i], repeated_coverage[i][0])

    #merges two lists together, creating pair combinations, each image is paired with a label
    def merge_lists(self, imgs, lbls):
        return [(imgs[i], lbls[i]) for i in range(0, len(imgs))]

"""
Main method to test operations worked, taking arugments from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #the name of the file that stores the images and corresponding coverages
    parser.add_argument("root_img_dir", help="the root image directory of the project")
    parser.add_argument("image_data_file", help="the name of the file containing the image paths and coverages")
    parser.add_argument("aug_folder", help="the name of the folder containing the augmented images")
    args = parser.parse_args()

    data_file_with_path = args.root_img_dir + args.image_data_file
    
    util = FileUtil(data_file_with_path)

    util.read_data()

    util.update_labels(args.aug_folder)
"""