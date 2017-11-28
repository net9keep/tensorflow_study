import tensorflow as tf
import matplotlib.pyplot as plt
x = [2,6,7]
y = [3,5,4.5]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.constant(0.0))

out = W*x+b

cost = tf.reduce_mean(tf.square(out-y))

a = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.initialize_all_variables()
s = tf.Session()
s.run(init)

for step in range(500):
    s.run(train)
    print(step, s.run(cost), s.run(W), s.run(b))

