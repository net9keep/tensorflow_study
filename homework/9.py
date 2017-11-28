import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
trainig_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
tf.random_normal([784, 300])
W1 = tf.get_variable('W1', shape=[784,300], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape=[300,300], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3', shape=[300,10], initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.Variable(tf.random_normal([300]))
B2 = tf.Variable(tf.random_normal([300]))
B3 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.matmul(X,W1) + B1)
L2 = tf.nn.relu(tf.matmul(L1,W2) + B2)
OUTPUT = tf.matmul(L2, W3) + B3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=OUTPUT, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(trainig_epochs):
    avg_cost = 0;
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X : batch_xs, Y : batch_ys})
        avg_cost += c/total_batch
    print(str(step) + '- cost : ' + str(avg_cost))

predicted = tf.equal(tf.argmax(OUTPUT,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(predicted,dtype=tf.float32))

print('accuracy : ',sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
