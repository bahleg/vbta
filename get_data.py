import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.ndimage.interpolation import rotate


def transpose(X):
    return np.array([x.reshape(28,28).T.flatten() for x in X])

def get_data(path='mnist'):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Labels_train = mnist.train.labels
    Labels_test = mnist.test.labels
    X_reverse_train =transpose(mnist.train.images)
    X_reverse_test = transpose(mnist.test.images)
    np.save(path, [X_train, X_test, X_reverse_train, X_reverse_test, Labels_train, Labels_test])



if __name__=='__main__':
    get_data()
