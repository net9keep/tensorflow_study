import tensorflow as tf
import matplotlib.pyplot as plt
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run([train, cost, W, b], feed_dict={X:[1,2,3], Y:[1,2,3]})
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

