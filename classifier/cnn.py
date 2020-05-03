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
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#initial learning rate
learn_rate = 0.001

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
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), padding='same', activation='relu',input_shape=(width, height, depth)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.2))
              
    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(nb_filters))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
             
    model.add(Dense(nb_filters))
    model.add(Activation('relu'))
    
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta(lr=learn_rate), metrics=['mean_absolute_error'])
    
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, number_epochs, path_to_checkpoint):
    # fit model  
    #checkpoint_path = path_to_checkpoint + "/seagrass_training/cp-{epoch:04d}.ckpt"
    checkpoint_path = path_to_checkpoint + "/seagrass_training/cp-seagrass.ckpt"

    # Create a callback that saves the model's weights
    # Saves every 5 epochs, only saving the latest as long as the file names is not unique
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor='val_mean_absolute_error', 
        save_weights_only=True, 
        save_best_only=True, 
        mode="max", 
        verbose=0)
    
    save_weights_to_disk(model, (checkpoint_path.format(epoch=0)))
    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=number_epochs, callbacks=[cp_callback])
    
    #print("\nHistory dict:", history.history)    
    
    return model

def evaluate_model(model, train_images, train_labels, test_images, test_labels):
    # evaluate the model
    train_mse = model.evaluate(train_images, train_labels, verbose=0)
    test_mse = model.evaluate(test_images, test_labels, verbose=0)
    
    _, acc = model.evaluate(test_images, test_labels, verbose=2)

    return train_mse, test_mse, acc

def plot_model_loss(history):
    # plot loss during training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
def plot_model_accuracy(history):
    #plot accuracy with epoch
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
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
    checkpoint_path = path + "/seagrass_training/cp-seagrass.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #checkpoints = [name for name in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, name))]
    #checkpoint_name = checkpoints[len(checkpoints)-1]
    #checkpoint_path += checkpoint_name
    
    #latest = tf.train.latest_checkpoint(checkpoint_path)
    model.load_weights(checkpoint_path)

    return model

def give_summary(model):
    model.summary()

def ini ():
    #if using a graph / InceptionNet, initialise here
    
    global cnn_instance
    cnn_instance = CNN()

def close():
    cnn_instance.close()
