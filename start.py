import tensorflow as tf
from cycleSR import CycleSR
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True, 'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('lambda1', 10.0, 'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_float('lambda2', 10.0, 'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float( 'pool_size', 50, 'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X_LR_INPUT', 'data/tfrecords/X_LR_INPUT.tfrecords', 'X tfrecords file for training, default: data/tfrecords/X_LR.tfrecords')
tf.flags.DEFINE_string('Y_LR_DIS', 'data/tfrecords/Y_LR_DIS.tfrecords', 'Y tfrecords file for training, default: data/tfrecords/Y_LR.tfrecords')
tf.flags.DEFINE_string('X_HR_DIS', 'data/tfrecords/X_HR_DIS.tfrecords', 'X tfrecords file for training, default: data/tfrecords/X_HR.tfrecords')
tf.flags.DEFINE_string('Y_HR_INPUT', 'data/tfrecords/Y_HR_INPUT.tfrecords', 'Y tfrecords file for training, default: data/tfrecords/Y_HR.tfrecords')

tf.flags.DEFINE_string(
    'load_model',
    None,
    'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + \
            FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():      # 设置默认的 图
        cycle_sr = CycleSR(
            input_ture_LR_x=FLAGS.X_LR_INPUT,

            input_ture_HR_y=FLAGS.Y_HR_INPUT,
            dis_true_HR_x=FLAGS.X_HR_DIS,
            dis_true_LR_y=FLAGS.Y_LR_DIS,
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, dis_fake_HR_x, dis_fake_LR_y = cycle_sr.model()

        optimizers = cycle_sr.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:     # 创建会话
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            dis_fake_HR_x_pool = ImagePool(FLAGS.pool_size)
            dis_fake_LR_y_pool = ImagePool(FLAGS.pool_size)

            while not coord.should_stop():
                # get previously generated images
                dis_fake_HR_x, dis_fake_LR_y = sess.run([dis_fake_HR_x, dis_fake_LR_y])

                # train
                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
                    sess.run(
                        [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                        feed_dict={cycle_sr.dis_fake_HR_x: dis_fake_HR_x_pool.query(dis_fake_HR_x),
                                   cycle_sr.dis_fake_LR_y: dis_fake_LR_y_pool.query(dis_fake_LR_y)}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 100 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  G_loss   : {}'.format(G_loss_val))
                    logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                    logging.info('  F_loss   : {}'.format(F_loss_val))
                    logging.info('  D_X_loss : {}'.format(D_X_loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(
                        sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(
                sess,
                checkpoints_dir +
                "/model.ckpt",
                global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
