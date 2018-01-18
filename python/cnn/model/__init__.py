#!/usr/bin/env python3.6

import os
import define
import summary
import functools
import tensorflow as tf

tf.nn.avg_pool = functools.partial(tf.nn.avg_pool, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
tf.layers.dense = functools.partial(tf.layers.dense, activation=tf.nn.relu)
tf.layers.conv2d = functools.partial(tf.layers.conv2d, filters=4, kernel_size=[4, 4], strides=[1, 1], padding="SAME")


def load_csv(file_name, number_of_columns=784, num_epochs=None):
    with tf.variable_scope("load_csv"):
        _, line = tf.TextLineReader(skip_header_lines=1).read(
            tf.train.string_input_producer([file_name], shuffle=False, num_epochs=num_epochs))
        features = tf.string_to_number(tf.string_split([line], ",").values, out_type=tf.int32)
        return tf.reshape(features, [-1, number_of_columns])


def export_submission(file_name, line):
    print_header = (os.path.isfile(file_name) is False)
    with open(file_name, "a+") as f:
        if print_header:
            f.write("ImageId,Label" + "\n")
        f.write(str(export_submission.i) + "," + str(line) + "\n")
        export_submission.i += 1


export_submission.i = 1

with tf.variable_scope("input"):
    X = tf.placeholder(tf.float32, [define.batch_size, 784])

with tf.variable_scope("output"):
    Y = tf.placeholder(tf.int32, [define.batch_size])
    output = tf.one_hot(Y, 10)


def create_cnn():
    input_layer = tf.reshape(X, [define.batch_size, 28, 28, 1])

    conv_layer_1_1 = tf.layers.conv2d(inputs=input_layer, filters=4, kernel_size=[2, 2])
    conv_layer_1_2 = tf.layers.conv2d(inputs=input_layer, filters=6, kernel_size=[3, 3])
    conv_layer_1_3 = tf.layers.conv2d(inputs=input_layer, filters=8, kernel_size=[4, 4])

    pooling_layer_1_1 = tf.nn.avg_pool(value=conv_layer_1_1)
    pooling_layer_1_2 = tf.nn.avg_pool(value=conv_layer_1_2)
    pooling_layer_1_3 = tf.nn.avg_pool(value=conv_layer_1_3)

    conv_layer_2_1 = tf.layers.conv2d(inputs=pooling_layer_1_1)
    conv_layer_2_2 = tf.layers.conv2d(inputs=pooling_layer_1_2)
    conv_layer_2_3 = tf.layers.conv2d(inputs=pooling_layer_1_3)

    pooling_layer_2_1 = tf.nn.avg_pool(value=conv_layer_2_1)
    pooling_layer_2_2 = tf.nn.avg_pool(value=conv_layer_2_2)
    pooling_layer_2_3 = tf.nn.avg_pool(value=conv_layer_2_3)

    to_vector_1 = tf.reshape(pooling_layer_2_1, [define.batch_size, -1])
    to_vector_2 = tf.reshape(pooling_layer_2_2, [define.batch_size, -1])
    to_vector_3 = tf.reshape(pooling_layer_2_3, [define.batch_size, -1])
    to_vector = tf.concat([to_vector_1, to_vector_2, to_vector_3], axis=1)

    dense_layer_1 = tf.layers.dense(inputs=to_vector,     units=300)
    dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=100)
    dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=10)

    return dense_layer_3


with tf.variable_scope("committee"):

    cnn_1 = create_cnn()
    cnn_2 = create_cnn()
    cnn_3 = create_cnn()
    cnn_4 = create_cnn()
    output_layer = tf.nn.softmax(tf.reduce_mean([cnn_1, cnn_2, cnn_3, cnn_4], 0))

with tf.variable_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=output, logits=output_layer))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.variable_scope("prediction"):
    prediction = tf.argmax(output_layer, 1)

with tf.variable_scope("evaluation"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(output, 1)), dtype=tf.float32))

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

x_test = load_csv(define.test_path, 784, 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

saver = tf.train.Saver()

# Summary
cost_summary = tf.summary.scalar("cost_train", cost)
summary_writer_train = tf.summary.FileWriter(summary.get_path("train"), tf.get_default_graph())
summary_writer_validation = tf.summary.FileWriter(summary.get_path("validation"), tf.get_default_graph())