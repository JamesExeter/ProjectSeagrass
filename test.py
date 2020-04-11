import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCH_SIZE = 5
TEST_SIZE = 0.2

def shuffle_data(images, labels):
    return shuffle(images, labels)

def batch(data, batch_size):
    # For item i in a range that is a length of l,
    for i in range(0, len(data), batch_size):
        # Create an index range for l of n items:
        yield data[i:i+batch_size]

def train_in_batch(images, labels, batch_size=BATCH_SIZE):
    images_batched = list(batch(images, batch_size))
    labels_batched = list(batch(labels, batch_size))

    #if a proper training set can't be made from the last batch, add the last batch to the one prior
    if len(images_batched[-1]) * TEST_SIZE < 1:
        last_images = images_batched[-1]
        last_labels = labels_batched[-1]
        del images_batched[-1]
        del labels_batched[-1]
        images_batched[-1] = np.append(images_batched[-1], last_images, axis=0)
        labels_batched[-1] = np.append(labels_batched[-1], last_labels, axis=0)

    print(images_batched, labels_batched)

    for i in range(len(images_batched)):
        print("Batch {}".format(i))
        
        train_images, test_images, train_labels, test_labels = train_test_split(images_batched[i], labels_batched[i], test_size=0.2, random_state=123)
        print(len(train_images), "Number of training images")
        print(len(test_images), "Number of test images")


if __name__ == "__main__":
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6,6 , 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    X,y = shuffle_data(X, y)

    train_in_batch(X, y)
