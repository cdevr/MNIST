import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import mnist
from tqdm import tqdm

print "getting training images"
train = mnist.train_images()

tf.reset_default_graph()

# Let's make an autoencoder image => [100] => image
# Then expand it into impage => [100] => [10] => [100] => image

# First level trainer
learning_rate = 2

img = tf.placeholder(tf.float32, [None, 28, 28])
flat_img = tf.reshape(img, [-1, 28*28], "flattened_image")

weight_l0 = tf.get_variable(shape=[28*28, 100], initializer=tf.contrib.layers.xavier_initializer(), name="weights_l0")
bias_l0 = tf.Variable(tf.random_normal([100]), name="bias_l0")
l0 = tf.nn.relu(tf.add(tf.matmul(flat_img, weight_l0), bias_l0))

weight_l5 = tf.Variable(tf.random_normal([100, 28*28]), name="weights_l5")
bias_l5 = tf.Variable(tf.random_normal([28*28]), name="bias_l5")
l5 = tf.add(tf.matmul(l0, weight_l5), bias_l5)

# Make an image out of the linear l5
l5 = tf.reshape(l5, [-1, 28, 28], name="output_reshape")

# For running example images through: no dropout
encodeDecode = tf.nn.relu(tf.add(tf.matmul(flat_img, weight_l0), bias_l0))
encodeDecode = tf.add(tf.matmul(encodeDecode, weight_l5), bias_l5)
encodeDecode = tf.reshape(encodeDecode, [-1, 28, 28])

cost = tf.reduce_mean(tf.squared_difference(l5, img))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

learned = {}
costs = []

sess = tf.Session(config=tf.ConfigProto())

sess.run(tf.global_variables_initializer())

batch_size = 20000
epochs = 400

# take 10 random images
drawImgs = random.sample(train, 10)

plt.ion()
plt.show()
plt.pause(1)

for epoch in tqdm(range(epochs), total=epochs):
    totcost, n = 0, 0

    batch_idx = 0
    while batch_idx < len(train):
        batch_end_idx = min(len(train), batch_idx + batch_size)
        batch_train = train[batch_idx: batch_end_idx]
        batch_idx += batch_size

        _, c = sess.run([optimizer, cost], feed_dict={img: batch_train})

        n += 1
        totcost += c
        costs.append(c)

    if epoch % 10 == 0:
        outImgs = sess.run(encodeDecode, feed_dict={img: drawImgs})
        for i, (draw, out) in enumerate(zip(drawImgs, outImgs)):
            plt.subplot(10, 2, 2*i+1)
            plt.imshow(draw)
            plt.subplot(10, 2, 2*i+2)
            plt.imshow(out)
        plt.show()
    plt.pause(0.01)

    if len(costs) > 30:
        tocheck = costs[-30:-20]
        if sum(tocheck)/len(tocheck) < costs[-1]:
            print "early stopping ! No cost decrease in 30 epochs: %2.2f -> %2.2f" % (sum(tocheck)/len(tocheck), costs[-1])
            break

    print "epoch %d: avg cost %2.2f last cost %2.2f" % (len(costs), totcost/n, c)

    if epoch % 10 == 0:
        learned['weight_l0'] = sess.run(weight_l0)
        learned['bias_l0'] = sess.run(bias_l0)
        learned['weight_l5'] = sess.run(weight_l5)
        learned['bias_l5'] = sess.run(bias_l5)

plt.ioff()
plt.show()