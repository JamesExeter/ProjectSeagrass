import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#initial learning rate
learn_rate = 0.001

cnn_instance = None

class CNN(object):
    #initialisation of CNN object, starting a tensorflow session
    def __init__(self):
        self.sess = tf.compat.v1.Session()
        self.lowest_loss = 1

    #closes the model instance
    def close(self):
        self.sess.close()

#creates the model given parameters of input width, height and depth
def create_cnn(width, height, depth):
    #variables that define the filters sizes
    nb_filters = 32
    nb_conv = 5
    nb_pool = 3
    
    model = Sequential()
    #input layer and conv_1
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), activation='relu', strides=2, padding='same', input_shape=(width, height, depth)))
    model.add(BatchNormalization())
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=3))
    
    #conv_2
    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=3))
        
    #conv_3
    model.add(Convolution2D(nb_filters*4, (nb_conv, nb_conv), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(nb_filters*4, (nb_conv, nb_conv), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=3))
    
    #global_average_pooling
    model.add(AveragePooling2D())
    model.add(Flatten())
    #regression output with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    #compile the model using mean squared error loss and adamdelta optimiser, using mean absolute error and mean squared error metrics
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learn_rate), metrics=['mean_absolute_error'])
    
    return model

#trains a model that have been initialised, takes the parameters of the model, the training and testing data
#the number of epochs to train for and the checkpoint path
def train_model(model, train_images, train_labels, test_images, test_labels, number_epochs, path_to_checkpoint):
    # fit model  
    #checkpoint_path = path_to_checkpoint + "/seagrass_training/cp-{epoch:04d}.ckpt"
    checkpoint_path = path_to_checkpoint + "/seagrass_training/cp-seagrass.ckpt"

    # Create a callback that saves the model's weights
    # Saves every 5 epochs, only saving the latest as long as the file names is not unique
    # or can only save the best which is now the case
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor='val_mean_absolute_error', 
        save_weights_only=True, 
        save_best_only=True, 
        mode="max", 
        verbose=0)
    
    #saves the weights at epoch 0 if better than last
    save_weights_to_disk(model, (checkpoint_path.format(epoch=0)))
    #starts the training of the model
    
    #stop the model earlier if it isn't improving
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=number_epochs, callbacks=[early_stop, cp_callback])
    
    #print("\nHistory dict:", history.history)    
    
    return model, history

#creates a history plotter object to use in other methods 
def create_history_plotter():
    return tfdocs.plots.HistoryPlotter(smoothing_std=2)

#evaluates a model using the training and test data, generating mean squared error and accuracy
def evaluate_model(model, test_images, test_labels):
    # evaluate the model
    mse, mae = model.evaluate(test_images, test_labels, verbose=2)

    return mse, mae

#plot the mae metric of the trained model
def plot_mae(history, plotter):
    plotter.plot({'Seagrass model' : history}, metric = 'mean_absolute_error')
    plt.ylim([0,10])
    plt.ylabel('MAE [MPG]')
    plt.show()

#plot the mse metric of the trained model
def plot_mse(history, plotter):
    plotter.plot({'Seagrass model' : history}, metric = 'mean_squared_error')
    plt.ylim([0,20])
    plt.ylabel('MSE [MPG^2]') 
    plt.show()   

#plots the predictions versus the actual values 
def plot_predictions_vs_actual(test_labels, test_predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('Ground-truth [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0,100]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    
def plot_prediction_error_distribution(test_labels, test_predictions):
    error = test_predictions - test_labels
    plt.hist(error, bins = 50)
    plt.xlabel('Prediction Error [MPG')
    _ = plt.ylabel("Count")
    plt.show()
    
#saves the entire model to file in a given location
#could allow the user to enter the name but may interrupt
#training is running model overnight
def save_model(model, save_path):
    name = "seagrass-model"
    model.save(save_path + "/" + name + ".h5")

#loads the most recently saved model from disk
def load_model_from_disk(load_path):
    #load the most recently saved model
    models = os.listdir(load_path)
    paths = [os.path.join(load_path, basename) for basename in models]
    latest_model = max(paths, key=os.path.getctime)
    
    new_model = load_model(latest_model)
    give_summary(new_model)
    
    return new_model

#saves the weights to disk given a path
def save_weights_to_disk(model, weights_path):
    model.save_weights(weights_path)

#used to load the latest saved weights from file
#especially useful during batch training
#this assumes only the best model weights are saved
#another approach would need to be used if saving weights periodically
#and the latest needed loading, e.g. find list of files in directory of 
#checkpoints, sort it and pick the last saved
def load_weights_from_disk(model, path):
    checkpoint_path = path + "/seagrass_training/cp-seagrass.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #checkpoints = [name for name in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, name))]
    #checkpoint_name = checkpoints[len(checkpoints)-1]
    #checkpoint_path += checkpoint_name
    
    #latest = tf.train.latest_checkpoint(checkpoint_path)
    model.load_weights(checkpoint_path)

    return model

#returns a summary of the network
def give_summary(model):
    model.summary()

#initialises the cnn instance
def ini ():
    #if using a graph / InceptionNet, initialise here
    #instead of creating a new instance, load the other model graph
    
    global cnn_instance
    cnn_instance = CNN()

#closes the cnn instance and its variables
def close():
    cnn_instance.close()