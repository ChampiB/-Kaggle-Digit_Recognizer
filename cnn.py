#!/usr/bin/env python3.6

import tensorflow as tf
import os.path
import summary
from tensorflow.examples.tutorials.mnist import input_data

##
## Functions
##
def load_csv(file_name, number_of_columns=784):

    with tf.variable_scope("load_csv"):
        _, line = tf.TextLineReader(skip_header_lines=1).read(tf.train.string_input_producer([file_name], shuffle=False, num_epochs=1))

        features = tf.string_to_number(tf.string_split([line], ",").values, out_type=tf.int32)

        return tf.reshape(features, [-1, number_of_columns])


def export_submission(file_name, predictions):

    print_header = (os.path.isfile(file_name) == False)

    with open(file_name, "a+") as f:
        if print_header:
            f.write("ImageId,Label" + "\n")
        f.write(str(export_submission.i) + "," + str(prediction) + "\n")
        export_submission.i += 1

export_submission.i = 1

##
## Model
##
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784])
    x = tf.nn.l2_normalize(x, axis=1)

with tf.variable_scope("dnn"):
    logits = tf.layers.dense(x,      300, activation=tf.nn.relu, name="input_layer")
    logits = tf.layers.dense(logits, 100, activation=tf.nn.relu, name="hidden_layer_1")
    logits = tf.layers.dense(logits, 10,  activation=tf.nn.relu, name="output_layer")
    logits = tf.nn.softmax(logits)

with tf.variable_scope("predict"):
    predict = tf.argmax(logits[0])

with tf.variable_scope("expected_output"):
    y = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

x_test = load_csv("./data/test.csv")

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

saver = tf.train.Saver()

# Summary
cost_train_summary      = tf.summary.scalar("cost_train", cost)
cost_validation_summary = tf.summary.scalar("cost_validation", cost)
summary_writer          = tf.summary.FileWriter(summary.get_path(), tf.get_default_graph())

with tf.Session() as sess:

    ##
    ## Training
    ##
    sess.run(init)

    if os.path.isfile("./save/cnn.ckpt.index") == True:
        saver.restore(sess, "./save/cnn.ckpt")

    for epoch in range(50001):
        batch_x, batch_y = mnist.train.next_batch(1000)
        _, error, cost_train_str = sess.run([optimizer, cost, cost_train_summary], feed_dict={x: batch_x, y: batch_y})
        if epoch % 100 == 0:
            print("Epoch : ", epoch, " Error : ", error)
            summary_writer.add_summary(cost_train_str, epoch)
            batch_x, batch_y = mnist.test.next_batch(1000)
            error, cost_validation_str = sess.run([cost, cost_validation_summary], feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(cost_validation_str, epoch)
        if epoch % 1000 == 0:
            saver.save(sess, "./save/cnn.ckpt")

    saver.save(sess, "./save/cnn.ckpt")

with tf.Session() as sess:

    ##
    ## Prediction
    ##
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver.restore(sess, "./save/cnn.ckpt")

    try:
        while True:
            prediction = sess.run(predict, feed_dict={x: sess.run(x_test)})
            export_submission("./submission.csv", prediction)
    except tf.errors.OutOfRangeError:
        print("Done !")

    ##
    ## Stop queue and threads
    ##
    coord.request_stop()
    coord.join(threads)
