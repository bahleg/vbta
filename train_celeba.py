import argparse
import tensorflow as tf
import numpy as np
from model import VBTA
from utils import callback, simple_batch_generator_build, make_dense, make_encoder_cnn, decoder_cnn
from functools import partial
import os
from get_data_celeba import get_data_celeba

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train celebA dataset')
    parser.add_argument('--model_save', type=str, default='./model', help='path to save model')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=10, help='epoch num')
    parser.add_argument('--t', type=float, default=1.0, help='value of triplet coef')
    parser.add_argument('--cc', type=float, default=0.0, help='value of cc coef')

    parser.add_argument('--seed', type=int, default=42, help='random seed for the experiment repeatability ')
    args = parser.parse_args()

    for required_data in ['femalex.npy', 'malex.npy']:
        if not os.path.exists(required_data):
            get_data_celeba()
    X_male = np.load('./malex.npy')
    X_female = np.load('./femalex.npy')

    X_male = X_male.reshape(X_male.shape[0], -1)  # flattening
    X_female = X_female.reshape(X_male.shape[0], -1)  # flattening
    X_train = X_male

    n_input = X_train.shape[1]
    n_latent = 64
    acitv = tf.nn.relu
    build_dict = {
        'encoder_x': make_encoder_cnn('encoder_x'),  # make_dense('encoder_x', n_hidden, acitv),
        'encoder_y': make_encoder_cnn('encoder_y'),  # make_dense('encoder_y', n_hidden,acitv),
        'encoder_common_mean': make_dense('encoder_common_mean', n_latent, None),
        'encoder_common_sigma': make_dense('encoder_common_sigma', n_latent, tf.nn.softplus),
        'decoder_same': decoder_cnn,

        'decoder_x_mean': make_dense('decoder_x_mean', n_input, None),
        'decoder_x_sigma': make_dense('decoder_x_sigma', n_input, tf.nn.softplus),
        'decoder_y_mean': make_dense('decoder_y_mean', n_input, None),
        'decoder_y_sigma': make_dense('decoder_y_sigma', n_input, tf.nn.softplus),
    }

    n_hidden = 512
    n_latent = 64
    acitv = tf.nn.relu
    n_input = X_train.shape[1]
    build_dict = {
        'encoder_x': make_dense('encoder_x', n_hidden, acitv),
        'encoder_y': make_dense('encoder_y', n_hidden, acitv),
        'encoder_common_mean': make_dense('encoder_common_mean', n_latent, None),
        'encoder_common_sigma': make_dense('encoder_common_sigma', n_latent, tf.nn.softplus),
        'decoder_same': make_dense('decoder_same', n_hidden, acitv),
        'decoder_x_mean': make_dense('decoder_x_mean', n_input, None),
        'decoder_y_mean': make_dense('decoder_y_mean', n_input, None),
    }

    model = VBTA(n_input, n_latent, build_dict, triplet_coef=args.t_start,
                 cc_coef=args.cc_start)
    rs = np.random.RandomState(args.seed)
    idx = range(X_train.shape[0])
    rs.shuffle(idx)
    semi = idx[:args.semi_supervised_size]

    batch_size = min(args.batch_size, args.semi_supervised_size)
    optimizer = tf.train.AdamOptimizer(10 ** (-3))

    batch_gen = simple_batch_generator_build(X_male, X_female, semi,
                                             batch_size,
                                             args.epoch_num,
                                             model)

    train_callback = partial(callback, X=X_male, Y=X_female, model_save=args.model_save)
    model.fit(optimizer, batch_gen, callback=train_callback, continue_train=False)
    model.save(args.model_save)

