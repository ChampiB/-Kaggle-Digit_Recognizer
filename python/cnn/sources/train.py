import os
import model
import logger
import define
import tensorflow as tf

#
# Training
#
print("### Start Training ###")
with tf.Session() as sess:
    sess.run(model.init)

    if os.path.isfile(define.save_path + ".index"):
        model.saver.restore(sess, define.save_path)

    for epoch in range(5001):
        batch_x, batch_y = model.mnist.train.next_batch(define.batch_size)
        if epoch % 100 == 0:
            print("Train set : ")
            logger.log_model_cost(sess, epoch, batch_x, batch_y, model.summary_writer_train)
            batch_x, batch_y = model.mnist.validation.next_batch(define.batch_size)
            print("Validation set : ")
            logger.log_model_cost(sess, epoch, batch_x, batch_y, model.summary_writer_validation)
            print("\n")
        else:
            sess.run([model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y})

        if epoch % 1000 == 0:
            model.saver.save(sess, define.save_path)

    model.saver.save(sess, define.save_path)

#
# Evaluate
#
print("### Start evaluation ###")
with tf.Session() as sess:
    sess.run(model.init)
    if os.path.isfile(define.save_path + ".index"):
        model.saver.restore(sess, define.save_path)
    epochs = 5
    total = 0
    for epoch in range(epochs):
        batch_x, batch_y = model.mnist.validation.next_batch(define.batch_size)
        total += sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
    total *= 100
    print("Accuracy : %.4f" % (total / epochs))
