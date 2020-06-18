import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import utils
import os
import gif
tf.compat.v1.disable_eager_execution()

def split_image(img):
    # We'll first collect all the positions in the image in our list, xs
    xs = []

    # And the corresponding colors for each of these positions
    ys = []

    # Now loop over the image
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # And store the inputs
            xs.append([row_i, col_i])
            # And outputs that the network needs to learn to predict
            ys.append(img[row_i, col_i])

    # we'll convert our lists to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def build_model(xs, ys, n_neurons, n_layers, activation_fn,
                final_activation_fn, cost_type):

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if xs.ndim != 2:
        raise ValueError(
            'xs should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
    if ys.ndim != 2:
        raise ValueError(
            'ys should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')

    n_xs = xs.shape[1]
    n_ys = ys.shape[1]

    X = tf.compat.v1.placeholder(name='X', shape=[None, n_xs],
                       dtype=tf.float32)
    Y = tf.compat.v1.placeholder(name='Y', shape=[None, n_ys],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = utils.linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = utils.linear(
        current_input, n_ys,
        activation=final_activation_fn,
        name='pred')[0]

    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(input_tensor=tf.reduce_sum(
                input_tensor=tf.abs(Y - Y_pred), axis=1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(input_tensor=tf.reduce_sum(
                input_tensor=tf.math.squared_difference(Y, Y_pred), axis=1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')

    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}


def train(imgs,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=10,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):

    N, H, W, C = imgs.shape
    all_xs, all_ys = [], []
    for img_i, img in enumerate(imgs):
        xs, ys = split_image(img)
        all_xs.append(np.c_[xs, np.repeat(img_i, [xs.shape[0]])])
        all_ys.append(ys)
    xs = np.array(all_xs).reshape(-1, 3)
    xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
    ys = np.array(all_ys).reshape(-1, 3)
    ys = ys / 127.5 - 1

    g = tf.Graph()
    with tf.compat.v1.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn,
                            cost_type)
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])
        sess.run(tf.compat.v1.global_variables_initializer())
        gifs = []
        costs = []
        step_i = 0
        for it_i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i * batch_size:
                              (batch_i + 1) * batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost

            print('iteration {}/{}: cost {}'.format(
                    it_i + 1, n_iterations, training_cost / n_batches))

            # Also, every 20 iterations, we'll draw the prediction of our
            # input xs, which should try to recreate our image!
            if (it_i + 1) % gif_step == 0:
                costs.append(training_cost / n_batches)
                ys_pred = model['Y_pred'].eval(
                    feed_dict={model['X']: xs}, session=sess)
                img = ys_pred.reshape(imgs.shape)
                gifs.append(img)
        return gifs

celeb_imgs = utils.get_celeb_imgs()
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(celeb_imgs).astype(np.uint8))
# It doesn't have to be 100 images, explore!
imgs = np.array(celeb_imgs).copy()
gifs = train(imgs=imgs)

montage_gifs = [np.clip(utils.montage(
            (m * 127.5) + 127.5), 0, 255).astype(np.uint8)
                for m in gifs]
_ = gif.build_gif(montage_gifs, saveto='multiple.gif')

final = gifs[-1]
final_gif = [np.clip(((m * 127.5) + 127.5), 0, 255).astype(np.uint8) for m in final]
gif.build_gif(final_gif, saveto='final.gif')
