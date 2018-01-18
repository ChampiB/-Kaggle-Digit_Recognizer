import model


def log_model_cost(sess, epoch, x, y, summary_writer):
    error, acc, cost_str = sess.run([model.cost, model.accuracy, model.cost_summary], feed_dict={model.X: x, model.Y: y})
    print("Epoch : %6s" % epoch, " Accuracy : %.4f" % (acc * 100), " Error : %.8f" % error)
    summary_writer.add_summary(cost_str, epoch)

