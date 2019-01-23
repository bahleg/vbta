import argparse
import tensorflow as tf
import numpy as np
from model import VBTA
from utils import callback, simple_batch_generator_build, make_dense
from functools import partial
import os
from get_data import get_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--data', type=str, default='./mnist.npy', help='location of the data array')
    parser.add_argument('--out_path', type=str, default='./eval.npy',
                        help='location of the data array for the evaulation')
    parser.add_argument('--model_save', type=str, default='./model', help='path to save model')
    parser.add_argument('--semi_supervised_size', type=int, default=10, help='size of labeled part of the dataset')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=10, help='epoch num')
    parser.add_argument('--t_start', type=float, default=10.0, help='initial value of triplet coef')
    parser.add_argument('--t_end', type=float, default=10.0, help='final value of triplet coef')
    parser.add_argument('--cc_start', type=float, default=1.0, help='initial value of cc coef')
    parser.add_argument('--cc_end', type=float, default=1.0, help='final value of cc coef')
    parser.add_argument('--seed', type=int, default=42, help='random seed for the experiment repeatability ')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        get_data(args.data)
    X_train, X_test, X_reverse_train, X_reverse_test, Labels_train, Labels_test = np.load(args.data)

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
    optimizer = tf.train.AdamOptimizer(10 ** (-4))
    batch_gen = simple_batch_generator_build(X_train, X_reverse_train, semi,
                                             batch_size,
                                             args.epoch_num,
                                             model,
                                             args.t_start,
                                             args.cc_start,
                                             args.t_end,
                                             args.cc_end, rs)

    train_callback = partial(callback, X=X_test, Y=X_reverse_test, model_save=args.model_save)
    model.fit(optimizer, batch_gen, callback=train_callback, continue_train=False)
    model.save(args.model_save)
    # data for evaluation
    z_x = model.latent_x(X_test, noise=False)
    to_test = [t.reshape(28, 28).T.flatten() for t in model.decode_z_y(z_x)]
    np.save(args.out_path, to_test)
