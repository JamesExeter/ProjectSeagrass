#!/usr/bin/env python

from os import listdir
import os
from PIL import Image
import cv2
import argparse

"""
Class used to check that the files loaded in a directory are indeed images and not another type of file
This class can be run by calling it with the command line by simply passing the path of the directory to check
"""
parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", help="Path to folder with images to check")
args = parser.parse_args()

for path, dirnames, filenames in os.walk(args.imgdir):
    for filename in filenames:
        if (filename.endswith('.jpg') or (filename.endswith('png'))):
            try:
                img_path = os.path.join(path, filename)
                img = Image.open(img_path) # open the image file
                img.verify() # verify that it is, in fact an image
                img = cv2.imread(img_path)
            except (IOError, SyntaxError) as e:
                print('Bad file:', img_path) # print out the names of corrupt files