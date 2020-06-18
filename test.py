import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import utils
import os
import gif
tf.compat.v1.disable_eager_execution()

def distance(p1,p2):
    return tf.abs(p1-p2)
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


X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,2], name='X')
h,W = utils.linear(X, 20, activation=tf.nn.relu)

f = os.path.join('img_paint','paint_img_2.jpg')
img = plt.imread(f)
square = utils.imcrop_tosquare(img)
img= np.array(Image.fromarray(square).resize((100,100), resample = Image.NEAREST))
# plt.figure(figsize=(5, 5))
# plt.imshow(img)
# plt.imsave(fname='reference.png', arr=img)
# plt.show()
xs, ys = split_image(img)
#Normalized the input
xs =  ((xs-np.mean(xs))/np.std(xs))
assert(np.min(xs) > -3.0 and np.max(xs) < 3.0)
ys = ys / 255.0
#reset graph
tf.compat.v1.reset_default_graph()

X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,2], name='X')
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,3], name='Y')
# We'll create 6 hidden layers.  Let's create a variable
# to say how many neurons we want for each of the layers
# (try 20 to begin with, then explore other values)
n_neurons = 40

h1, W1 = utils.linear(X, n_neurons, activation=tf.nn.sigmoid, name='layer1')
h2, W2 = utils.linear(h1, n_neurons, activation=tf.nn.sigmoid, name='layer2')
h3, W3 = utils.linear(h2, n_neurons, activation=tf.nn.sigmoid, name='layer3')
h4, W4 = utils.linear(h3, n_neurons, activation=tf.nn.sigmoid, name='layer4')
h5, W5 = utils.linear(h4, n_neurons, activation=tf.nn.sigmoid, name='layer5')
h6, W6 = utils.linear(h5, n_neurons, activation=tf.nn.sigmoid, name='layer6')

# Now, make one last layer to make sure your network has 3 outputs:
Y_pred, W8 = utils.linear(h3, 3, activation=tf.nn.sigmoid, name='pred')

#Computed error, summed all the error and found the mean to use as cost
error = distance(Y_pred,Y)
sum_error = tf.reduce_sum(error,1)
# cost = tf.reduce_mean(sum_error)
cost = tf.reduce_mean(tf.reduce_sum(
                tf.compat.v1.squared_difference(Y, Y_pred), 1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=.01).minimize(cost)
n_iterations = 1500
batch_size = 50
sess = tf.compat.v1.Session()

# Initialize all your variables and run the operation with your session
sess.run(tf.compat.v1.global_variables_initializer())

# Optimize over a few iterations, each time following the gradient
# a little at a time
imgs = []
costs = []
gif_step = n_iterations // 10
step_i = 0

for it_i in range(n_iterations):

    # Get a random sampling of the dataset
    idxs = np.random.permutation(range(len(xs)))

    # The number of batches we have to iterate over
    n_batches = len(idxs) // batch_size

    # Now iterate over our stochastic minibatches:
    for batch_i in range(n_batches):

        # Get just minibatch amount of data
        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        # And optimize, also returning the cost so we can monitor
        # how our optimization is doing.
        training_cost = sess.run(
            [cost, optimizer],
            feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]

    # Also, every 20 iterations, we'll draw the prediction of our
    # input xs, which should try to recreate our image!
    if (it_i + 1) % gif_step == 0:
        costs.append(training_cost / n_batches)
        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
        img = np.clip(ys_pred.reshape(img.shape), 0, 1)
        imgs.append(img)
        # Plot the cost over time
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(costs)
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Cost')
        ax[1].imshow(img)
        fig.suptitle('Iteration {}'.format(it_i))
        plt.show()

_ = gif.build_gif(imgs, saveto='single.gif', show_gif=False)
