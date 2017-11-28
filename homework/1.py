import tensorflow as tf
import matplotlib.pyplot as plt

def TwoDimensionGraph():
    x_data = [2, 6, 7]
    y_data = [3, 5, 4.5]

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    hypothesis = W * x_data

    cost = tf.reduce_sum(tf.pow(hypothesis-y_data, 2)) / len(x_data)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    W_data = []
    cost_data = []
    for i in range(-40, 40):
        temp = i * 0.1
        W_data.append(temp)
        cost_data.append(sess.run(cost, feed_dict={W:W_data[i+40]}))
    sess.close()

    #graph
    plt.plot(W_data, cost_data)
    plt.xlabel('W')
    plt.ylabel('COST')
    plt.show()

def minimize():
    x_data = [2, 6, 7]
    y_data = [3, 5, 4.5]

    W = tf.Variable(tf.random_uniform([1]))
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    hypothesis =  W * x
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)
    reduce_mean = tf.reduce_mean((W * x - y) * x)
    gradient = W - 0.01 * reduce_mean
    update = W.assign(gradient)
    init_tf = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init_tf)

    count = 0

    for step in range(2000):
        count += 1
        update_data = sess.run(update, feed_dict={x : x_data, y : y_data})
        cost_data = sess.run(cost, feed_dict={x : x_data, y : y_data})
        W_data = sess.run(W)
        reduce_mean_data = sess.run(reduce_mean, feed_dict={x : x_data, y : y_data})
        if count == 20:
            print('step['+str(step+1)+'] W : ' + str(W_data) + ' cost : ' + str(cost_data))
            count = 0


minimize()

# x_data = [1, 2, 3]
# y_data = [1, 2, 3]
#
# W = tf.Variable(tf.random_uniform([1]))
#
# X = tf.placeholder(tf.float32, name="X")
# Y = tf.placeholder(tf.float32, name="Y")
#
# print(X)
# print(Y)
#
# hypothesis = W*X+b
#
# cost = tf.reduce_mean(tf.square(hypothesis-Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train_op = optimizer.minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(100):
#         _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
#
#         print(step, cost_val, sess.run(W), sess.run(b))
#
#     print("\n=== Test ===")
#     print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
#     print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
#
#

