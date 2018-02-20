#!/usr/bin/env python3.6

import tensorflow as tf
import os.path
import summary
import define
from tensorflow.examples.tutorials.mnist import input_data

##
## Functions
##
def load_csv(file_name, number_of_columns=784, num_epochs=None):

    with tf.variable_scope("load_csv"):
        _, line = tf.TextLineReader(skip_header_lines=1).read(tf.train.string_input_producer([file_name], shuffle=False, num_epochs=num_epochs))
        features = tf.string_to_number(tf.string_split([line], ",").values, out_type=tf.int32)
        return tf.reshape(features, [-1, number_of_columns])

def export_submission(file_name, prediction):

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

with tf.variable_scope("expected_output"):
    y = tf.placeholder(tf.int64, [None, 10])

with tf.variable_scope("dnn"):
    logits = tf.layers.dense(x,      300, activation=tf.nn.relu, name="input_layer")
    logits = tf.layers.dense(logits, 100, activation=tf.nn.relu, name="hidden_layer_1")
    logits = tf.layers.dense(logits, 10,  activation=tf.nn.relu, name="output_layer")
    logits = tf.nn.softmax(logits)

with tf.variable_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.variable_scope("prediction"):
    prediction = tf.argmax(logits, 1)

with tf.variable_scope("evalutation"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, 1)), dtype=tf.float32))

x_test  = load_csv(define.test_path,  784, 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

saver = tf.train.Saver()

# Summary
cost_train_summary      = tf.summary.scalar("cost_train", cost)
cost_validation_summary = tf.summary.scalar("cost_validation", cost)
summary_writer          = tf.summary.FileWriter(summary.get_path(), tf.get_default_graph())

##
## Training
##
print("### Start Training ###")
with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if os.path.isfile(define.save_path + ".index") == True:
        saver.restore(sess, define.save_path)

    for epoch in range(2001):
        batch_x, batch_y = mnist.train.next_batch(1000)
        if epoch % 100 == 0:
            _, error, cost_train_str = sess.run([optimizer, cost, cost_train_summary], feed_dict={x: batch_x, y: batch_y})
            print("Epoch : ", epoch, " Error : ", error)
            summary_writer.add_summary(cost_train_str, epoch)
        else:
            sess.run([optimizer, cost, cost_train_summary], feed_dict={x: batch_x, y: batch_y})

        if epoch % 1000 == 0:
            saver.save(sess, define.save_path)

    saver.save(sess, define.save_path)

    coord.request_stop()
    coord.join(threads)

##
## Evaluate
##
print("### Start evalutation ###")
with tf.Session() as sess:

    sess.run(init)
    if os.path.isfile(define.save_path + ".index") == True:
        saver.restore(sess, define.save_path)
    print("Accuracy : ", sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels}) * 100)

##
## Prediction
##
print("### Start prediction ###")
with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver.restore(sess, define.save_path)

    try:
        while True:
            y_test = sess.run(prediction, feed_dict={x: sess.run(x_test)})
            export_submission(define.export_path, int(y_test))
    except tf.errors.OutOfRangeError:
        print("Done !")

    coord.request_stop()
    coord.join(threads)
