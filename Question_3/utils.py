import numpy as np
from keras.datasets import mnist

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

def load_data():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape the data to 2D
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    return X_train, y_train, X_test, y_test
