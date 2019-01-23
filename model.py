import tensorflow as tf
import numpy as np
from utils import jensen_shannon_approximate


class VBTA_semi(object):
    """
    Variational Bi-domain Triplet Autoencoder (for semi-supervised regime)
    """

    def __init__(self, n_input, n_latent, build_dict, triplet_coef=1.0, cc_coef=1.0):
        """
        :param n_input: input dimension of the data
        :param n_latent:  latent dimension
        :param build_dict:  dictionary, containing constructors for the submodels (see train.py)
        :param triplet_coef: initial value for the triplet coef.
        :param cc_coef: initial value for the cc coef.
        """
        self.n_input = n_input
        self.n_latent = n_latent

        self.triplet_coef = tf.placeholder_with_default(np.array(triplet_coef).astype(np.float32), ())
        self.cc_coef = tf.placeholder_with_default(np.array(cc_coef).astype(np.float32), ())

        self.is_t_from_y = tf.placeholder(tf.float32, [])
        # encoder for x
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.pre_z_x = build_dict['encoder_x'](self.x)
        self.mean_x_enc = build_dict['encoder_common_mean'](self.pre_z_x)
        self.sigma_sq_x_enc = build_dict['encoder_common_sigma'](self.pre_z_x)

        # encoder for x semi
        self.x_semi = tf.placeholder(tf.float32, [None, self.n_input])
        self.pre_z_x_semi = build_dict['encoder_x'](self.x_semi)
        self.mean_x_semi_enc = build_dict['encoder_common_mean'](self.pre_z_x_semi)
        self.sigma_sq_x_semi_enc = build_dict['encoder_common_sigma'](self.pre_z_x_semi)

        # encoder for y
        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        self.pre_z_y = build_dict['encoder_y'](self.y)
        self.mean_y_enc = build_dict['encoder_common_mean'](self.pre_z_y)
        self.sigma_sq_y_enc = build_dict['encoder_common_sigma'](self.pre_z_y)

        # encoder for y semi
        self.y_semi = tf.placeholder(tf.float32, [None, self.n_input])
        self.pre_z_y_semi = build_dict['encoder_y'](self.y_semi)
        self.mean_y_semi_enc = build_dict['encoder_common_mean'](self.pre_z_y_semi)
        self.sigma_sq_y_semi_enc = build_dict['encoder_common_sigma'](self.pre_z_y_semi)

        # encoder for t
        self.t = tf.placeholder(tf.float32, [None, self.n_input])
        # case: T is from Y
        self.pre_z_ty = build_dict['encoder_y'](self.t)
        self.mean_ty_enc = build_dict['encoder_common_mean'](self.pre_z_ty)
        self.sigma_sq_ty_enc = build_dict['encoder_common_sigma'](self.pre_z_ty)

        # case: T is from X
        self.pre_z_tx = build_dict['encoder_x'](self.t)
        self.mean_tx_enc = build_dict['encoder_common_mean'](self.pre_z_tx)
        self.sigma_sq_tx_enc = build_dict['encoder_common_sigma'](self.pre_z_tx)

        # sample from gaussian distribution
        eps_x = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_latent]), 0, 1, dtype=tf.float32)
        eps_y = tf.random_normal(tf.stack([tf.shape(self.y)[0], self.n_latent]), 0, 1, dtype=tf.float32)
        self.z_x = tf.add(self.mean_x_enc, tf.multiply(tf.sqrt(self.sigma_sq_x_enc), eps_x))
        self.z_y = tf.add(self.mean_y_enc, tf.multiply(tf.sqrt(self.sigma_sq_y_enc), eps_y))

        self.z_x_semi = tf.add(self.mean_x_semi_enc, tf.multiply(tf.sqrt(self.sigma_sq_x_semi_enc), eps_x))
        self.z_y_semi = tf.add(self.mean_y_semi_enc, tf.multiply(tf.sqrt(self.sigma_sq_y_semi_enc), eps_y))

        # decoder for x, reconstruction_x is from common space
        self.reconstruction_x = build_dict['decoder_same'](self.z_x)
        self.reconstruction_x_semi = build_dict['decoder_same'](self.z_x_semi)

        self.mean_x_dec = build_dict['decoder_x_mean'](self.reconstruction_x)

        self.mean_xy_dec = build_dict['decoder_y_mean'](self.reconstruction_x_semi)

        # decoder for y, reconstruction_y is from common space
        self.reconstruction_y = build_dict['decoder_same'](self.z_y)
        self.reconstruction_y_semi = build_dict['decoder_same'](self.z_y_semi)

        self.mean_y_dec = build_dict['decoder_y_mean'](self.reconstruction_y)

        self.mean_yx_dec = build_dict['decoder_x_mean'](self.reconstruction_y_semi)

        # cost by x

        rec_x = 0.5 * tf.reduce_sum((tf.pow(tf.subtract(self.mean_x_dec, self.x), 2.0)), axis=1)

        rec_yx = 0.5 * tf.reduce_sum((tf.pow(tf.subtract(self.mean_yx_dec, self.x_semi), 2.0)), axis=1)

        logpxz = rec_x + rec_yx * self.cc_coef

        KDL_x = -0.5 * tf.reduce_sum(1 + tf.log(self.sigma_sq_x_enc)
                                     - tf.pow(self.mean_x_enc, 2.0)
                                     - self.sigma_sq_x_enc, 1)

        # cost by y
        rec_y = 0.5 * tf.reduce_sum((tf.pow(tf.subtract(self.mean_y_dec, self.y), 2.0)), axis=1)

        rec_xy = 0.5 * tf.reduce_sum((tf.pow(tf.subtract(self.mean_xy_dec, self.y_semi), 2.0)), axis=1)

        logpyz = rec_y + rec_xy * self.cc_coef

        KDL_y = -0.5 * tf.reduce_sum(1 + tf.log(self.sigma_sq_y_enc)
                                     - tf.pow(self.mean_y_enc, 2.0)
                                     - self.sigma_sq_y_enc, 1)

        # D_YT when T is from X
        D_yt = jensen_shannon_approximate(self.mean_y_semi_enc, self.mean_tx_enc,
                                          tf.sqrt(self.sigma_sq_y_semi_enc),
                                          tf.sqrt(self.sigma_sq_tx_enc))

        # D_XT when T is from Y
        D_xt = jensen_shannon_approximate(self.mean_x_semi_enc, self.mean_ty_enc,
                                          tf.sqrt(self.sigma_sq_x_semi_enc),
                                          tf.sqrt(self.sigma_sq_ty_enc))

        D_xy = jensen_shannon_approximate(self.mean_x_semi_enc, self.mean_y_semi_enc,
                                          tf.sqrt(self.sigma_sq_x_semi_enc),
                                          tf.sqrt(self.sigma_sq_y_semi_enc))

        self.D_xt = D_xt
        self.D_yt = D_yt
        self.D_xy = D_xy

        try:

            triplet_kl_loss_t_from_y = -((tf.nn.log_softmax([D_xt, D_xy], axis=0)[0]))
            triplet_kl_loss_t_from_x = -((tf.nn.log_softmax([D_yt, D_xy], axis=0)[0]))
        except:
            print 'using old version of softmax'
            triplet_kl_loss_t_from_y = -((tf.nn.log_softmax([D_xt, D_xy], dim=0)[0]))
            triplet_kl_loss_t_from_x = -((tf.nn.log_softmax([D_yt, D_xy], dim=0)[0]))

        self.pure_triplet_loss = self.is_t_from_y * tf.reduce_mean(triplet_kl_loss_t_from_y) + (
                                                                                                   1.0 - self.is_t_from_y) * tf.reduce_mean(
            triplet_kl_loss_t_from_x)
        self.triplet_loss = triplet_coef * self.pure_triplet_loss
        self.cost = tf.reduce_mean(KDL_x + logpxz + KDL_y + logpyz) + self.triplet_coef * tf.reduce_mean(
            self.triplet_loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.losses = [rec_x,
                       rec_yx,
                       rec_y,
                       rec_yx,
                       KDL_x, KDL_y, self.pure_triplet_loss]

        self.train_op = None

    def save(self, fname):
        saver = tf.train.Saver()
        saver.save(self.sess, fname)

    def load(self, fname):
        saver = tf.train.Saver()
        saver.restore(self.sess, fname)

    def calc_total_cost(self, X, Y, T, t_from_y=1.0):
        return self.sess.run([self.cost], feed_dict={self.x: X, self.y: Y, self.x_semi: X,
                                                     self.y_semi: Y, self.t: T, self.is_t_from_y: t_from_y})

    def latent_x(self, X, noise=True):
        """
        Encoder target object with encoder-x
        :param X:  object
        :param noise:  use sampling or mean
        :return: z_x (or mean of its distribution)
        """
        if noise:
            return self.sess.run(self.z_x, feed_dict={self.x: X})
        else:
            return self.sess.run(self.mean_x_enc, feed_dict={self.x: X})

    def latent_y(self, Y, noise=True):
        """
        Encoder target object with encoder-x
        :param Y:  object
        :param noise:  use sampling or mean
        :return: z_y (or mean of its distribution)
        """
        if noise:
            return self.sess.run(self.z_y, feed_dict={self.y: Y})
        else:
            return self.sess.run(self.mean_y_enc, feed_dict={self.y: Y})

    def decode_z_x(self, z_x):
        return self.sess.run(self.mean_x_dec, feed_dict={self.z_x: z_x})

    def decode_z_y(self, z_y):
        return self.sess.run(self.mean_y_dec, feed_dict={self.z_y: z_y})

    def reconstruct_x(self, X):
        return self.sess.run(self.mean_x_dec, feed_dict={self.x: X})

    def reconstruct_y(self, Y):
        return self.sess.run(self.mean_y_dec, feed_dict={self.y: Y})

    def mean_x_encode(self, X):
        return self.sess.run(self.mean_x_enc, feed_dict={self.x: X})

    def mean_y_encode(self, Y):
        return self.sess.run(self.mean_y_enc, feed_dict={self.y: Y})

    def sigma_x_encode(self, X):
        return self.sess.run(self.sigma_sq_x_enc, feed_dict={self.x: X})

    def sigma_y_encode(self, Y):
        return self.sess.run(self.sigma_sq_y_enc, feed_dict={self.y: Y})

    def fit(self, optimizer, batch_gen, callback=None, continue_train=False):
        """
        Optimization procedure
        :param optimizer: optimizer
        :param batch_gen: generator of dataset batches
        :param callback:  callback function (can be None)
        :param continue_train: if True and train op was preciously initialized, does not restart session
        :return:
        """
        if not self.train_op or continue_train is False:
            train_op = optimizer.minimize(self.cost)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.train_op = train_op

        for opt_info, batch_feed_dict in batch_gen:
            self.sess.run(self.train_op, feed_dict=batch_feed_dict)
            if callback is not None:
                callback(opt_info, self)


class VBTA(object):
    """
       Variational Bi-domain Triplet Autoencoder
       """

    def __init__(self, n_input, n_latent, build_dict, triplet_coef=1.0, cc_coef=1.0):
        """
        :param n_input: input dimension of the data
        :param n_latent:  latent dimension
        :param build_dict:  dictionary, containing constructors for the submodels (see train.py)
        :param triplet_coef: value for the triplet coef.
        :param cc_coef: value for the cc coef.
        """
        self.n_input = n_input
        self.n_latent = n_latent
        kl_divergence = tf.distributions.kl_divergence
        norm_distr = tf.distributions.Normal

        self.is_t_from_y = tf.placeholder(tf.float32, [])
        # encoder for x
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.pre_z_x = build_dict['encoder_x'](self.x)
        self.mean_x_enc = build_dict['encoder_common_mean'](self.pre_z_x)
        self.sigma_sq_x_enc = build_dict['encoder_common_sigma'](self.pre_z_x)

        # encoder for y
        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        self.pre_z_y = build_dict['encoder_y'](self.y)
        self.mean_y_enc = build_dict['encoder_common_mean'](self.pre_z_y)
        self.sigma_sq_y_enc = build_dict['encoder_common_sigma'](self.pre_z_y)

        # encoder for t, where t looks more like y than x
        self.t = tf.placeholder(tf.float32, [None, self.n_input])
        # case: T from Y
        self.pre_z_ty = build_dict['encoder_y'](self.t)
        self.mean_ty_enc = build_dict['encoder_common_mean'](self.pre_z_ty)
        self.sigma_sq_ty_enc = build_dict['encoder_common_sigma'](self.pre_z_ty)

        # case: T from X
        self.pre_z_tx = build_dict['encoder_x'](self.t)
        self.mean_tx_enc = build_dict['encoder_common_mean'](self.pre_z_tx)
        self.sigma_sq_tx_enc = build_dict['encoder_common_sigma'](self.pre_z_tx)

        # sample from gaussian distribution
        eps_x = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_latent]), 0, 1, dtype=tf.float32)
        eps_y = tf.random_normal(tf.stack([tf.shape(self.y)[0], self.n_latent]), 0, 1, dtype=tf.float32)
        self.z_x = tf.add(self.mean_x_enc, tf.multiply(tf.sqrt(self.sigma_sq_x_enc), eps_x))
        self.z_y = tf.add(self.mean_y_enc, tf.multiply(tf.sqrt(self.sigma_sq_y_enc), eps_y))

        # decoder for x, reconstruction_x is from common space
        self.reconstruction_x = build_dict['decoder_same'](self.z_x)
        self.mean_x_dec = build_dict['decoder_x_mean'](self.reconstruction_x)

        self.mean_xy_dec = build_dict['decoder_y_mean'](self.reconstruction_x)

        # decoder for y, reconstruction_y is from common space
        self.reconstruction_y = build_dict['decoder_same'](self.z_y)
        self.mean_y_dec = build_dict['decoder_y_mean'](self.reconstruction_y)
        self.mean_yx_dec = build_dict['decoder_x_mean'](self.reconstruction_y)

        rec_x = 0.5 * tf.reduce_sum(((tf.pow(tf.subtract(self.mean_x_dec, self.x), 2.0))), axis=1)

        rec_yx = 0.5 * tf.reduce_sum((tf.pow(tf.subtract(self.mean_yx_dec, self.x), 2.0)), axis=1)

        logpxz = rec_x + (rec_yx) * cc_coef

        KDL_x = -0.5 * tf.reduce_sum(1 + tf.log(self.sigma_sq_x_enc)
                                     - tf.pow(self.mean_x_enc, 2.0)
                                     - self.sigma_sq_x_enc, 1)

        # cost by y
        rec_y = 0.5 * tf.reduce_sum(((tf.pow(tf.subtract(self.mean_y_dec, self.y), 2.0))), axis=1)

        rec_xy = 0.5 * tf.reduce_sum(((tf.pow(tf.subtract(self.mean_xy_dec, self.y), 2.0))), axis=1)

        logpyz = rec_y + (rec_xy) * cc_coef

        KDL_y = -0.5 * tf.reduce_sum(1 + tf.log(self.sigma_sq_y_enc)
                                     - tf.pow(self.mean_y_enc, 2.0)
                                     - self.sigma_sq_y_enc, 1)

        distr_q_x = norm_distr(loc=self.mean_x_enc, scale=self.sigma_sq_x_enc + 0.01)
        distr_q_y = norm_distr(loc=self.mean_y_enc, scale=self.sigma_sq_y_enc + 0.01)

        # D_YT when T is from X
        D_yt = jensen_shannon_approximate(self.mean_y_enc, self.mean_tx_enc,
                                          tf.sqrt(self.sigma_sq_y_enc),
                                          tf.sqrt(self.sigma_sq_tx_enc))

        # D_XT when T is from Y
        D_xt = jensen_shannon_approximate(self.mean_x_enc, self.mean_ty_enc,
                                          tf.sqrt(self.sigma_sq_x_enc),
                                          tf.sqrt(self.sigma_sq_ty_enc))

        D_xy = jensen_shannon_approximate(self.mean_x_enc, self.mean_y_enc,
                                          tf.sqrt(self.sigma_sq_x_enc),
                                          tf.sqrt(self.sigma_sq_y_enc))

        self.D_xt = D_xt
        self.D_yt = D_yt
        self.D_xy = D_xy

        try:

            triplet_kl_loss_t_from_y = -((tf.nn.log_softmax([D_xt, D_xy], axis=0)[0]))
            triplet_kl_loss_t_from_x = -((tf.nn.log_softmax([D_yt, D_xy], axis=0)[0]))
        except:

            triplet_kl_loss_t_from_y = -((tf.nn.log_softmax([D_xt, D_xy], dim=0)[0]))
            triplet_kl_loss_t_from_x = -((tf.nn.log_softmax([D_yt, D_xy], dim=0)[0]))

        self.triplet_loss = triplet_coef * self.is_t_from_y * tf.reduce_mean(
            triplet_kl_loss_t_from_y) + triplet_coef * (1.0 - self.is_t_from_y) * tf.reduce_mean(
            triplet_kl_loss_t_from_x)

        self.cost = tf.reduce_mean(KDL_x + logpxz + KDL_y + logpyz)

        self.cost = self.cost + triplet_coef * tf.reduce_mean(self.triplet_loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        self.sess.run(init)

        self.losses = []

        self.train_op = None

    def save(self, fname):
        saver = tf.train.Saver()
        saver.save(self.sess, fname)

    def load(self, fname):
        saver = tf.train.Saver()
        saver.restore(self.sess, fname)

    def calc_total_cost(self, X, Y, T, train_plc, t_from_y=1.0):
        return self.sess.run([self.cost], feed_dict={self.x: X, self.y: Y, self.t: T, self.is_t_from_y: t_from_y,
                                                     train_plc: False})

    def latent_x(self, X, noise=True):
        if noise:
            return self.sess.run(self.z_x, feed_dict={self.x: X})
        else:
            return self.sess.run(self.mean_x_enc, feed_dict={self.x: X})

    def latent_y(self, Y, noise=True):
        if noise:
            return self.sess.run(self.z_y, feed_dict={self.y: Y})
        else:
            return self.sess.run(self.mean_y_enc, feed_dict={self.y: Y})

    def decode_z_x(self, z_x):
        return self.sess.run(self.mean_x_dec, feed_dict={self.z_x: z_x})

    def decode_z_y(self, z_y):
        return self.sess.run(self.mean_y_dec, feed_dict={self.z_y: z_y})

    def reconstruct_x(self, X):
        return self.sess.run(self.mean_x_dec, feed_dict={self.x: X})

    def reconstruct_y(self, Y):
        return self.sess.run(self.mean_y_dec, feed_dict={self.y: Y})

    def mean_x_encode(self, X):
        return self.sess.run(self.mean_x_enc, feed_dict={self.x: X})

    def mean_y_encode(self, Y):
        return self.sess.run(self.mean_y_enc, feed_dict={self.y: Y})

    def sigma_x_encode(self, X):
        return self.sess.run(self.sigma_sq_x_enc, feed_dict={self.x: X})

    def sigma_y_encode(self, Y):
        return self.sess.run(self.sigma_sq_y_enc, feed_dict={self.y: Y})

    def fit(self, optimizer, batch_gen, callback=None, continue_train=False):
        """
        Optimization procedure
        :param optimizer: optimizer
        :param batch_gen: generator of dataset batches
        :param callback:  callback function (can be None)
        :param continue_train: if True and train op was preciously initialized, does not restart session
        :return:
        """
        if not self.train_op or continue_train is False:
            train_op = optimizer.minimize(self.cost)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.train_op = train_op

        for iter_id, batch_feed_dict in batch_gen:
            self.sess.run(self.train_op, feed_dict=batch_feed_dict)
            if callback is not None:
                callback(iter_id, self)
