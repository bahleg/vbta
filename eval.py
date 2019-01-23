import argparse
import tensorflow as tf
import numpy as np
from model import VBTA
from utils import callback, simple_batch_generator_build, make_dense
from functools import partial
import os
from get_data import get_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--data', type=str, default='./mnist.npy', help='location of the data array')
    parser.add_argument('--eval_path', type=str, default='./eval.npy',
                        help='location of the data array for the evaulation')
    args = parser.parse_args()

    """
    classifier model
    """
    import os

    GPUID = 1
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
    # Parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    display_step = 1


    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


    b_conv1 = bias_variable([32])
    W_conv1 = weight_variable([5, 5, 1, 32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess=sess, save_path="./model/conv_classify.ckpt")
    """
    Evaluation
    """
    X_train, X_test, X_reverse_train, X_reverse_test, Labels_train, Labels_test = np.load(args.data)
    if not os.path.exists(args.data):
        get_data(args.data)

    to_classify = [t for t in np.load(args.eval_path)]
    acc = (sess.run(accuracy, feed_dict={x: to_classify, y_: Labels_test, keep_prob: 1.0}))
    print 'Accuracy', acc
