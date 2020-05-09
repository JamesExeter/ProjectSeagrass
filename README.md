# ProjectSeagrass
A CNN regression project that aims to give coverage estimates for seagrass images
This project was built within a virtual environment on Ubuntu 18 that uses Python 3.6.9, Tensorflow 1.14, Keras 2.3.1, NumPy 1.18.2, Scikit-learn 0.22.2.post1 and OpenCV 4.2.0

This projects covers the formatting process of images for usage in the CNN model
Seagrass quadrat images are cropped to remove the quadrat from the image and processed to include a formatted naming scheme
Various feature extractions have been tested on images to discern what features of interest the neural network can learn

The main program can be run by using bash scripts on Linux or Windows using a bash command executor. The program can be run
without the need to use bash scripts but due to the required command lines arguments it is advantageous to use a script.

If the script is set for training, it will load the selected formatted dataset and then train the model and then evaluate 
its performance. A model will be generated following this training that is stored for later loading.

Given a model is trained for use, a directory of images that are formatted to be of the shape 576x576x3 can be passed to the
trained model with which coverage estimates are generated and stored in the results folder. 

To make predictions on a set of custom images that adhere to the required image format (top-down quadrat image).

Using the make_predictions.sh script, when prompted if the images need formatting first, select yes and then follow the prompts.
The main method is called which crops and resizes the images. Then, given that a model is trained and saved, predictions
will be made and saved to a file in the folder you chose. 