import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os
from reader import Reader

REAL_LABEL = 0.9

"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""


class CycleSR(object):

    def __init__(
            self,
            input_ture_LR_x='',
            input_ture_HR_y='',
            dis_true_HR_x='',
            dis_true_LR_y='',
            batch_size=1,
            image_size=256,   #ture LR image size
            use_lsgan=True,
            norm='instance',
            lambda1=10.0,
            lambda2=10.0,
            learning_rate=2e-4,
            beta1=0.5,
            ngf=64,
            img_size=100,
            num_layers=32,
            feature_size=256,
            scale=2,
            output_channels=3):
        """

        :param img_size:
        :param num_layers:
        :param feature_size:
        :param scale:
        :param output_channels:

        lambda1: integer, weight for forward cycle loss (X->Y->X)
        lambda2: integer, weight for backward cycle loss (Y->X->Y)

        """
        print("Building EDSR...")
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.input_ture_LR_x = input_ture_LR_x
        self.input_ture_HR_y = input_ture_HR_y
        self.dis_true_HR_x = dis_true_HR_x
        self.dis_true_LR_y = dis_true_LR_y
        self.batch_size = batch_size
        self.img_size = img_size
        self.scale = scale
        self.hr_img_size = image_size*scale
        self.output_channels = output_channels

        # Placeholder for image inputs
        self.ture_LR = x = tf.placeholder(
            tf.float32, [None, img_size, img_size, output_channels])
        # Placeholder for upscaled image ground-truth
        self.ture_HR = y = tf.placeholder(
            tf.float32, [
                None, img_size * scale, img_size * scale, output_channels])

        self.fake_x = tf.placeholder(tf.float32,
                                     shape=[None, img_size*scale, img_size*scale, 3])
        self.fake_y = tf.placeholder(tf.float32,
                                     shape=[None, img_size*scale, img_size*scale, 3])
        # Placeholder for image inputs
        # LR_X & HR_Y
        # self.input_ture_HR_y = tf.placeholder(
        #     tf.float32, [None, img_size * scale, img_size * scale, output_channels])
        # self.input_ture_LR_x = tf.placeholder(
        #     tf.float32, [None, img_size, img_size, output_channels])

        # Placeholder for generated fake image to compute discriminate loss and generated loss
        self.dis_fake_HR_x = tf.placeholder(
            tf.float32, [None, img_size * scale, img_size * scale, output_channels])
        self.dis_fake_LR_y = tf.placeholder(
            tf.float32, [None, img_size, img_size, output_channels])

        # Placeholder for image ground-truth to compute discriminate loss and generated loss
        # self.dis_true_HR_x = tf.placeholder(
        #     tf.float32, [None, img_size * scale, img_size * scale, output_channels])
        # self.dis_true_LR_y = tf.placeholder(
        #     tf.float32, [None, img_size, img_size, output_channels])

        # Placeholder for generated images by fake images to compute cycle-consistency loss
        self.cyc_fake_HR_y = tf.placeholder(
            tf.float32, [None, img_size * scale, img_size * scale, output_channels])
        self.cyc_fake_LR_x = tf.placeholder(
            tf.float32, [None, img_size, img_size, output_channels])

        # image_input_lr, mean_x= self.preprossessing(x)



        # self.loss = loss = tf.reduce_mean(
        #     tf.losses.absolute_difference(
        #         self.input_HR, output))



        # Scalar to keep track for loss
        # tf.summary.scalar("g_loss", self.g_loss)
        # tf.summary.scalar("loss", self.loss)
        # tf.summary.scalar("g_loss", self.loss)
        # tf.summary.scalar("loss", self.loss)
        # tf.summary.scalar("PSNR", PSNR)
        # Image summaries for input, target, and output
        # tf.summary.image("input_image", tf.cast(self.ture_LR, tf.uint8))
        # tf.summary.image("target_image", tf.cast(self.ture_HR, tf.uint8))
        # tf.summary.image("output_image", tf.cast(self.out, tf.uint8))

        # Tensorflow graph setup... session, saver, etc.
        # self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        print("Done building!")

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )

            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step)
            )
            return learning_step

        G_optimizer = make_optimizer(G_loss, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, name='Adam_D_Y')
        F_optimizer = make_optimizer(F_loss, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, name='Adam_D_X')

        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')

    """
    build model
    """
    def model(self):

        X_LR_reader = Reader(self.input_ture_LR_x, name='X',
                             image_size=int(self.img_size/2), batch_size=self.batch_size)
        Y_HR_reader = Reader(self.input_ture_HR_y, name='Y',
                             image_size=self.img_size, batch_size=self.batch_size)
        X_HR_reader = Reader(self.dis_true_HR_x, name='X',
                             image_size=self.img_size, batch_size=self.batch_size)
        Y_LR_reader = Reader(self.dis_true_LR_y, name='Y',
                             image_size=int(self.img_size/2), batch_size=self.batch_size)

        input_ture_LR_x = X_LR_reader.feed()
        input_ture_HR_y = Y_HR_reader.feed()
        dis_true_HR_x = X_HR_reader.feed()
        dis_true_LR_y = Y_LR_reader.feed()

        # gen_G fn, gen_F fn, input ture LR_X, input ture HR_y
        cycle_loss = self.cyc_loss(self.gen_G, self.gen_F, input_ture_LR_x, input_ture_HR_y)

        # X_LR -> Y_HR
        dis_fake_HR_x = self.gen_G(feature_size=256, num_layers=32, scale=2, x=input_ture_LR_x)
        G_gan_loss = self.g_loss(self.dis_Y, dis_fake_HR_x)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss = self.d_loss(self.dis_Y, dis_true_HR_x, self.dis_fake_HR_x)  # use unpaired ? ? ?

        # Y_HR -> X_LR
        dis_fake_LR_y = self.gen_F(64, 32, 2, input_ture_HR_y)
        F_gan_loss = self.g_loss(self.dis_X, dis_fake_LR_y)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = self.d_loss(self.dis_X, dis_true_LR_y, self.dis_fake_LR_y)  # use unpaired ? ? ?

        # dis_fake_HR_x = self.dis_fake_HR_x
        # dis_fake_LR_y = self.dis_fake_LR_y

        return G_loss, D_Y_loss, F_loss, D_X_loss, dis_fake_HR_x, dis_fake_LR_y
    """
    Preprocessing as mentioned in the paper, by subtracting the mean
    		However, the subtract the mean of the entire dataset they use. As of
    		now, I am subtracting the mean of each batch
    """

    def preprossessing(self, x, img_size):
        mean_x = 127  # tf.reduce_mean(self.input)
        x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])

        image_input = x - mean_x

        return image_input, mean_x

    """
    Generator G
    x: LR -> y: HR
    x = tf.placeholder(tf.float32, [None, img_size, img_size, output_channels])
    """
    def gen_G(self, feature_size, num_layers, scale, x):
        image_input, mean_x = self.preprossessing(x, int(self.img_size/2))
        # One convolution before res blocks and to convert to required feature
        # depth
        # conv  # input ( 32*32 ) output ( 32*32 )
        x = slim.conv2d(image_input, feature_size, [3, 3])

        # Store the output of the first convolution to add later      *********
        conv_1 = x

        """
        This creates `num_layers` number of resBlocks
        a resBlock is defined in the paper as
        (excuse the ugly ASCII graph)
        x
        |\
        | \
        |  conv2d
        |  relu
        |  conv2d
        | /
        |/
        + (addition here)
        |
        result
        """

        """
        Doing scaling here as mentioned in the paper:

        `we found that increasing the number of feature
        maps above a certain level would make the training procedure
        numerically unstable. A similar phenomenon was
        reported by Szegedy et al. We resolve this issue by
        adopting the residual scaling with factor 0.1. In each
        residual block, constant scaling layers are placed after the
        last convolution layers. These modules stabilize the training
        procedure greatly when using a large number of filters.
        In the test phase, this layer can be integrated into the previous
        convolution layer for the computational efficiency.'

        """
        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(num_layers):  # 32        conv_1---conv_64
            x = utils.resBlock(x, feature_size, scale=scaling_factor)

        # One more convolution, and then we add the output of our first conv
        # layer      *******************
        # conv_65                       #   LR -> HR
        x = slim.conv2d(x, feature_size, [3, 3])
        x += conv_1  # 补齐残差

        # Upsample output of the convolution
        x = utils.upsample(x, scale, feature_size, None)  # conv_66 conv_67
        x = tf.clip_by_value(x + mean_x, 0.0, 255.0)


        return x

    """
    Generator_F   
    HR -> LR
    |
    conv
    |\
    | \ 
    |  \
    |   conv
    |   relu
    |   conv
    |   mul
    |  /
    | /
    |/
    conv
    downsample
    |
    """
    def gen_F(self, feature_size, num_layers, scale, y):
        image_input, mean_x = self.preprossessing(y, self.img_size)
        # One convolution before res blocks and to convert to required feature
        # depth
        # conv  # input x2 ( 64*64 ) output ( 64*64 ) 64
        y = slim.conv2d(image_input, feature_size, [3, 3])

        # Store the output of the first convolution to add later      *********
        conv_1 = y

        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(num_layers):  # 32        conv_1---conv_64
            y = utils.resBlock(y, feature_size, scale=scaling_factor)

        # One more convolution, and then we add the output of our first conv
        # layer      *******************
        # conv_65                       #   HR -> LR
        y = slim.conv2d(y, feature_size, [3, 3])
        y += conv_1  # 补齐残差

        # Downsample output of the convolution
        y = utils.downsample(y, scale, feature_size, None)  # conv_66 conv_67 scale = 2
        y = tf.clip_by_value(y + mean_x, 0.0, 255.0)

        return y

    """
    discriminator_X
    """
    def dis_X(self, feature_size, x):  # feature_size = 64 as default
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
        activation = lrelu
        # L0 k3n64s1
        x = slim.conv2d(x, feature_size, [9, 9], activation_fn=activation)
        # L1 k3n64s1
        x = slim.conv2d(x, feature_size, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L2 k3n128s1
        x = slim.conv2d(x, feature_size * 2, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L3 k3n128s1
        x = slim.conv2d(x, feature_size * 2, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L4 k3n256s1
        x = slim.conv2d(x, feature_size * 4, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L5 k3n256s1
        x = slim.conv2d(x, feature_size * 4, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L6 k3n512s1
        x = slim.conv2d(x, feature_size * 8, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L7 k3n512s1
        x = slim.conv2d(x, feature_size * 8, [3, 3], activation_fn=activation)
        x = slim.batch_norm(x, activation_fn=lrelu, is_training=True, param_initializers=None)
        # Flatten layer
        x = slim.flatten(x)
        # Dense (1024)
        x = slim.fully_connected(x, num_outputs=1024, activation_fn=lrelu)
        # Dense (1024)
        # x = tf.layers.dense(x, units=1024, activation=lrelu, name="dense_1024", reuse=tf.AUTO_REUSE)
        # Dense (1)
        x = slim.fully_connected(x, 1, activation_fn=lrelu)
        # x = tf.layers.dense(x, units=1, activation=lrelu, name="dense_1")

        x = tf.nn.sigmoid(x, name="out")

        return x

    """
        discriminator_Y
        """

    def dis_Y(self, feature_size, y):  # feature_size = 64 as default
        lrelu = lambda y: tf.nn.leaky_relu(y, 0.2)
        activation = lrelu
        # L0 k3n64s1
        y = slim.conv2d(y, feature_size, [9, 9], activation_fn=activation)
        # L1 k3n64s1
        y = slim.conv2d(y, feature_size, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L2 k3n128s1
        y = slim.conv2d(y, feature_size * 2, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L3 k3n128s1
        y = slim.conv2d(y, feature_size * 2, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L4 k3n256s1
        y = slim.conv2d(y, feature_size * 4, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L5 k3n256s1
        y = slim.conv2d(y, feature_size * 4, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L6 k3n512s1
        y = slim.conv2d(y, feature_size * 8, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)
        # L7 k3n512s1
        y = slim.conv2d(y, feature_size * 8, [3, 3], activation_fn=activation)
        y = slim.batch_norm(y, activation_fn=lrelu, is_training=True, param_initializers=None)

        # Flatten layer
        y = slim.flatten(y)
        # Dense (1024)
        y = slim.fully_connected(y, 1024, activation_fn=lrelu)
        # Dense (1)
        y = slim.fully_connected(y, 1, activation_fn=lrelu)

        y = tf.nn.sigmoid(y, name="out")

        return y


    """
    Discriminator loss
    """
    def d_loss(self, Dis, ture_data, false_data):  # Dis: discriminator_fn
        error_real = tf.reduce_mean(tf.squared_difference(Dis(64, ture_data), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(Dis(64, false_data)))

        loss = (error_fake + error_real)/2



        return loss

    """
    Generator loss
    """
    def g_loss(self, Dis, false_data):
        # loss = tf.reduce_mean(tf.squared_difference(Dis(64, false_data), REAL_LABEL))

        loss = tf.reduce_mean(0.9)
        return loss

    """
    Cycle consistency loss
    """
    def cyc_loss(self, G, F, ture_LR, ture_HR):
        cX_loss = tf.reduce_mean(tf.abs(F(64, 32, 2, G(256, 32, 2, ture_LR))-ture_LR))   # gen_G(self, feature_size, num_layers, scale, x):
        cY_loss = tf.reduce_mean(tf.abs(G(256, 32, 2, F(64, 32, 2, ture_HR))-ture_HR))
        loss = self.lambda1*cX_loss + self.lambda2*cY_loss
        return loss

    """
    final_output
    
    """
    def final_out(self, x, mean_x):
        # One final convolution on the upsampling output
        output = x  # slim.conv2d(x,output_channels,[3,3])
        self.out = tf.clip_by_value(output + mean_x, 0.0, 255.0)
        return x


    """
	Save the current state of the network to file
	"""

    def save(self, savedir='saved_models'):
        print("Saving...")
        self.saver.save(self.sess, savedir + "/model")
        print("Saved!")

    """
	Resume network from previously saved weights
	"""

    def resume(self, savedir='saved_models'):
        print("Restoring...")
        self.saver.restore(self.sess, tf.train.latest_checkpoint(savedir))
        print("Restored!")




    """
    PSNR
    """
    def psnr(self, input_HR, input_LR):
        # Calculating Peak Signal-to-noise-ratio
        # Using equations from here:
        # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        mse = tf.reduce_mean(tf.squared_difference(input_HR, self.gen_G(input_LR)))
        PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * utils.log10(PSNR)
        return PSNR


