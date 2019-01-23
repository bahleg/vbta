import tensorflow as tf
import numpy as np
from functools import partial
import tensorflow.contrib.slim as slim


# Thanks to https://github.com/LynnHo/VAE-Tensorflow/blob/master/models_64x64.py
def flatten_fully_connected(inputs,
                            num_outputs,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None):
    with tf.variable_scope(scope, 'flatten_fully_connected', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)


conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu


def callback(optimization_info, model, X, Y, model_save):
    """
    Simple callback function to monitor optimization process.
    Every epoch calculates cost.
    Uses subsample fo test-data for evaluation and default values for triplet and cc coefs.
    :param optimization_info: dict
    :param model: model object
    :param X: first domain test data
    :param Y: second domain test data
    """
    if optimization_info['iteration'] == 0 and optimization_info['epoch'] % 1 == 0:
        ind = (range(X.shape[0]))
        np.random.shuffle(ind)
        ind = ind[:50]
        print 'Epoch', optimization_info['epoch'], model.calc_total_cost(X[ind], Y[ind],
                                                                         Y[ind[::-1]])

    if optimization_info['iteration'] == 0 and optimization_info['epoch'] > 0:
        print 'saving'
        model.save(model_save + '_' + str(optimization_info['epoch']))


def gau_kl(pm, pv, qm, qv):
    """
    Thanks to https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1. / qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)  # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)  # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis)  # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))


def js_approximate_np(mu_a, mu_b, sigma_a, sigma_b):
    dists = np.zeros((mu_a.shape[0], mu_b.shape[0]))
    for i in xrange(mu_a.shape[0]):
        for j in xrange(mu_b.shape[0]):
            dists[i, j] = 0.5 * (gau_kl(mu_a[i], sigma_a[i] ** 2, mu_b[j], sigma_b[j] ** 2)) + \
                          0.5 * (gau_kl(mu_b[j], sigma_b[j] ** 2, mu_a[i], sigma_a[i] ** 2))
    return dists


def js_np_batch_generator_build(data_x, data_y, semi_indices, batch_size, training_epochs, model):
    n_samples = data_x.shape[0]
    indices = range(n_samples)

    total_batch = int(n_samples / batch_size)
    for epoch in range(training_epochs):

        for i in range(total_batch):
            np.random.shuffle(indices)
            np.random.shuffle(semi_indices)
            # semi_indices = indices

            batch_x = data_x[indices[:batch_size]]
            batch_y = data_y[indices[:batch_size]]

            batch_x_semi = data_x[semi_indices[:batch_size]]
            batch_y_semi = data_y[semi_indices[:batch_size]]

            mean_x = model.mean_x_encode(batch_x_semi)
            mean_y = model.mean_y_encode(batch_y_semi)
            sigma_x = model.sigma_x_encode(batch_x_semi)
            sigma_y = model.sigma_y_encode(batch_y_semi)

            js_distances = js_approximate_np(mean_x, mean_y, sigma_x, sigma_y)

            np.fill_diagonal(js_distances, 99999)

            batch_t = np.zeros(batch_x.shape)
            for q in xrange(0, mean_x.shape[0]):
                batch_t[q] = batch_y_semi[np.argmin(js_distances[q])]

            # zloebach = semi_indices[:batch_size]
            # np.random.shuffle(zloebach)
            # batch_t = data_y[zloebach]

            yield {'epoch': epoch, 'iteration': i, }, {model.x: batch_x, model.y: batch_y,
                                                       model.x_semi: batch_x_semi,
                                                       model.y_semi: batch_y_semi,
                                                       model.t: batch_t}


def simple_batch_generator_build(data_x, data_y, batch_size, training_epochs, model):
    """
    Batch generator
    :param data_x: Training data for X domain
    :param data_y: Training data for Y domain
    :param batch_size: int
    :param training_epochs: int
    :param model: model to train
    :return: generator of meta-data and feed_dict
    """
    n_samples = data_x.shape[0]
    indices = list(range(n_samples))
    total_batch = int(n_samples / batch_size)
    for epoch in list(range(training_epochs)):
        for i in list(range(total_batch)):
            np.random.shuffle(indices)

            batch_x = data_x[indices[:batch_size]]
            batch_y = data_y[indices[:batch_size]]

            random_batch_idx = indices[:batch_size]
            np.random.shuffle(random_batch_idx)

            t_from_y = i % 2 == 0

            if t_from_y:
                batch_t = data_y[random_batch_idx]
            else:
                batch_t = data_x[random_batch_idx]

            yield {'epoch': epoch, 'iteration': i, }, {model.x: batch_x, model.y: batch_y,
                                                       model.t: batch_t, model.is_t_from_y: float(t_from_y)}


def simple_batch_generator_build_semi(data_x, data_y, semi_indices, batch_size, training_epochs, model, t_start,
                                      c_start,
                                      t_end,
                                      c_end, rs, fixed_epochs=2):
    """
    Batch generator for semi-supervised learning
    :param data_x: Training data for X domain
    :param data_y: Training data for Y domain
    :param semi_indices: indices of data for semi-supervised learning
    :param batch_size: int
    :param training_epochs: int
    :param model: model to train
    :param t_start: initial value for triplet coef
    :param c_start: initial value for cc coef
    :param t_end: final value for triplet coef
    :param c_end: final value for triplet coef
    :param rs: Random State
    :param fixed_epochs: epochs without coefficient annealing
    :return: generator of meta-data and feed_dict
    """
    n_samples = data_x.shape[0]
    indices = range(n_samples)
    total_batch = int(n_samples / batch_size)
    full_iter_num = total_batch * (training_epochs - fixed_epochs)
    cur_id = -1
    start_id = total_batch * fixed_epochs

    for epoch in range(training_epochs):
        for i in range(total_batch):
            cur_id += 1

            if epoch < fixed_epochs:
                cur_t = t_start
                cur_c = c_start
            else:
                cur_t = t_start + (t_end - t_start) * (cur_id - start_id) / full_iter_num
                cur_c = c_start + (c_end - c_start) * (cur_id - start_id) / full_iter_num

            rs.shuffle(indices)
            rs.shuffle(semi_indices)

            batch_x = data_x[indices[:batch_size]]
            batch_y = data_y[indices[:batch_size]]

            batch_x_semi = data_x[semi_indices[:batch_size]]
            batch_y_semi = data_y[semi_indices[:batch_size]]

            random_batch_idx = semi_indices[:batch_size]
            rs.shuffle(random_batch_idx)

            t_from_y = i % 2 == 0

            if t_from_y:
                batch_t = data_y[random_batch_idx]
            else:
                batch_t = data_x[random_batch_idx]

            yield {'epoch': epoch, 'iteration': i, }, {model.x: batch_x, model.y: batch_y, model.x_semi: batch_x_semi,
                                                       model.y_semi: batch_y_semi,
                                                       model.t: batch_t, model.is_t_from_y: float(t_from_y),
                                                       model.triplet_coef: cur_t, model.cc_coef: cur_c}


def jensen_shannon_approximate(mu_a, mu_b, sigma_sq_a, sigma_sq_b):
    """
    function for JS approximation.
    """
    kl_divergence = tf.distributions.kl_divergence
    norm_distr = tf.distributions.Normal
    distr_a = norm_distr(loc=mu_a, scale=sigma_sq_a + 0.01)
    distr_b = norm_distr(loc=mu_b, scale=sigma_sq_b + 0.01)
    kl_ab = 0.5 * kl_divergence(distr_a, distr_b, allow_nan_stats=False)
    kl_ba = 0.5 * kl_divergence(distr_b, distr_a, allow_nan_stats=False)
    distance = (tf.reduce_sum((kl_ab + kl_ba), 1))
    return distance


def make_dense(name, output_unit_num, activation):
    """
    Simple wrapper for Dense layer
    """
    return tf.layers.Dense(units=output_unit_num, activation=activation,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name=name)


dim = 64


def make_encoder_cnn(name):
    def encoder(img):
        conv_bn_lrelu = partial(conv, activation_fn=relu, biases_initializer=None)

        with tf.variable_scope('encoder' + name, reuse=tf.AUTO_REUSE):
            y = tf.reshape(img, [-1, 64, 64, 3])
            y = conv_bn_lrelu(y, dim, 5, 2)
            y = conv_bn_lrelu(y, dim * 2, 5, 2)
            # y = conv_bn_lrelu(y, dim * 4, 5, 2)
            # y = conv_bn_lrelu(y, dim * 8, 5, 2)
            return tf.contrib.layers.flatten(y)

    return encoder


def decoder_cnn(z):
    dconv_bn_relu = partial(dconv, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        y = fc(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = relu(y)
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        # y = dconv_bn_relu(y, dim * 2, 5, 2)
        # y = dconv_bn_relu(y, dim * 1, 5, 2)
        y = dconv(y, 3, 5, 2)
        return tf.contrib.layers.flatten(y)
