import tensorflow as tf
import numpy as np


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


def simple_batch_generator_build(data_x, data_y, semi_indices, batch_size, training_epochs, model, t_start, c_start,
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
