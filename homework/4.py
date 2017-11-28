import tensorflow as tf
import math
import numpy as np

INPUT_COUNT=2
OUTPUT_COUNT=2
HIDDEN_COUNT=2
LEARNING_RATE=0.4
MAX_STEPS=5000

INPUT_TRAIN = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OUTPUT_TRAIN = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

input_placeholder = tf.placeholder("float", shape=[None, INPUT_COUNT])
output_placeholder = tf.placeholder("float", shape=[None, OUTPUT_COUNT])

feed_dict = {
    input_placeholder : INPUT_TRAIN,
    output_placeholder : OUTPUT_TRAIN
}

weight_hidden = tf.Variable(tf.truncated_normal([INPUT_COUNT, HIDDEN_COUNT]))
bias_hidden = tf.Variable(tf.zeros([HIDDEN_COUNT]))
af_hidden = tf.nn.sigmoid(tf.matmul(input_placeholder, weight_hidden) + bias_hidden)
weight_output = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, OUTPUT_COUNT]))
bias_output = tf.Variable(tf.zeros([OUTPUT_COUNT]))


logits = tf.matmul(af_hidden, weight_output) + bias_output
y = tf.nn.softmax(logits)


cross_entropy = -tf.reduce_sum(output_placeholder * tf.log(y))
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

predicted = tf.cast(logits > 0, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, output_placeholder), dtype=tf.float32))

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(MAX_STEPS):
        value = sess.run([train_step, loss], feed_dict)
        if i % 100 == 0:
            print("---------"+str(i)+"-----------")
            print("value" + str(value))
            print("----------------------------")
            for input_value in INPUT_TRAIN:
                print input_value, sess.run(y, feed_dict={input_placeholder:[input_value]})
            print(sess.run([predicted, accuracy], feed_dict))