import model
import define
import tensorflow as tf

#
# Prediction
#
print("### Start prediction ###")
with tf.Session() as sess:
    sess.run(model.init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    model.saver.restore(sess, define.save_path)

    try:
        while True:
            y_test = sess.run(model.prediction, feed_dict={model.X: sess.run(model.x_test)})
            model.export_submission(define.export_path, int(y_test))
    except tf.errors.OutOfRangeError:
        print("Done !")

    coord.request_stop()
    coord.join(threads)
