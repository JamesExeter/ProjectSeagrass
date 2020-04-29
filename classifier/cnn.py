import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D, BatchNormalization
from keras.optimizers import Adadelta
from keras.models import load_model
from keras_tqdm import TQDMNotebookCallback
import matplotlib.pyplot as pyplot

cnn_instance = None

class CNN(object):
    def __init__(self):
        self.sess = tf.compat.v1.Session()

    def close(self):
        self.sess.close()

def create_cnn(width, height, depth):
    nb_filters = 32
    nb_conv = 5
    nb_pool = 3

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, strides=nb_conv,
                            input_shape=(width, height, depth), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=None, padding='valid', data_format=None)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
              
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=None, padding='valid', data_format=None)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
              
    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=None, padding='valid', data_format=None)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(nb_filters*2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
             
    model.add(Dense(nb_filters))
    model.add(Activation('relu'))
    
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta(), metrics=['mean_absolute_error'])
    
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, number_epochs, path_to_checkpoint):
    # fit model  
    checkpoint_path = path_to_checkpoint + "/seagrass_training/cp-{epoch:04d}.ckpt"

    # Create a callback that saves the model's weights
    # Saves every 5 epochs, only saving the latest as long as the file names is not unique 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_weights_only=True, save_best_only=False, mode="auto", verbose=0, period=5)
    
    save_weights_to_disk(model, (checkpoint_path.format(epoch=0)))
    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=number_epochs, verbose=0, callbacks=[cp_callback, TQDMNotebookCallback(verbose=2)])
    
    return history

def evaluate_model(model, train_images, train_labels, test_images, test_labels):
    # evaluate the model
    train_mse = model.evaluate(train_images, train_labels, verbose=0)
    test_mse = model.evaluate(test_images, test_labels, verbose=0)
    
    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Model accuracy: {:5.2f}%".format(100*acc))

    return train_mse, test_mse

def plot_training_results(history):
    # plot loss during training
    pyplot.title('Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    
def save_model(model, save_path):
    name = "seagrass-model"
    model.save(save_path + "/" + name + ".h5")

def load_model_from_disk(load_path):
    #load the most recently saved model
    models = os.listdir(load_path)
    paths = [os.path.join(load_path, basename) for basename in models]
    latest_model = max(paths, key=os.path.getctime)
    
    new_model = load_model(latest_model)
    give_summary(new_model)
    
    return new_model

def save_weights_to_disk(model, weights_path): 
    model.save_weights(weights_path)

def load_weights_from_disk(model, path):
    checkpoint_path = "seagrass_training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(path + "/" + checkpoint_path)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    model.load_weights(latest)
    
    return model

def give_summary(model):
    model.summary()

def ini ():
    #if using a graph / InceptionNet, initialise here
    
    global cnn_instance
    cnn_instance = CNN()
    

def close():
    cnn_instance.close()
