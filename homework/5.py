import tensorflow as tf
import math
import numpy as np

INPUT_COUNT = 2
OUTPUT_COUNT = 1
HIDDEN_COUNT = 2
LEARNING_RATE = 0.1
MAX_STEPS = 5000

INPUT_TRAIN = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OUTPUT_TRAIN = np.array([[0], [1], [1], [0]])

inputs_placeholder = tf.placeholder("float", shape=[None, INPUT_COUNT])
labels_placeholder = tf.placeholder("float", shape=[None, OUTPUT_COUNT])

feed_dict = {
    inputs_placeholder: INPUT_TRAIN,
    labels_placeholder: OUTPUT_TRAIN,
}

WEIGHT_HIDDEN = tf.Variable(tf.truncated_normal([INPUT_COUNT, HIDDEN_COUNT]))
BIAS_HIDDEN = tf.Variable(tf.zeros([HIDDEN_COUNT]))

AF_HIDDEN = tf.nn.sigmoid(tf.matmul(inputs_placeholder, WEIGHT_HIDDEN) + BIAS_HIDDEN)

WEIGHT_OUTPUT = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, OUTPUT_COUNT]))
BIAS_OUTPUT = tf.Variable(tf.zeros([OUTPUT_COUNT]))

logits = tf.matmul(AF_HIDDEN, WEIGHT_OUTPUT) + BIAS_OUTPUT
y = tf.nn.sigmoid(logits)

loss = -tf.reduce_mean(labels_placeholder*tf.log(y)+(1-labels_placeholder)*tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

predicted = tf.cast(logits > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels_placeholder), dtype=tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(MAX_STEPS):

        sess.run(train_step, feed_dict)

        if step % 100 == 0:
            print(step, sess.run(loss, feed_dict))

    print(sess.run([predicted, accuracy], feed_dict))

    print "\nCorrect: ", c, "\nAccuracy: ", a