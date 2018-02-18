from __future__ import division

import tensorflow as tf
import numpy as np
import os
from .ops import *
from .util import *
from progressbar import ETA, Bar, Percentage, ProgressBar

class STGConvnet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.net_type = config.net_type
        self.resize_size = (110, 200)
        self.action_size = 3
        self.batch_size = config.batch_size
        self.dense_layer = config.dense_layer
        self.num_frames = config.num_frames
        self.num_chain = config.num_chain
        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.beta1 = config.beta1
        self.step_size = config.step_size
        self.sample_steps = config.sample_steps

        self.data_path = os.path.join(config.data_path) #, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)
        self.log_dir = os.path.join(self.output_dir, 'log')
        self.train_dir = os.path.join(self.output_dir, 'observed_sequence')
        self.sample_dir = os.path.join(self.output_dir, 'synthesis_sequence')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.result_dir = os.path.join(self.output_dir, 'final_result')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
    def descriptor(self, inputs, reuse=False, input_action = None, dense_layer = True):
        with tf.variable_scope('des', reuse=reuse):
            if self.net_type == 'STG_3':
                """
                STG_action V0.4 20180217
                """
                conv1 = conv3d(inputs, 60, (3, 7, 7), strides=(1, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 60, (3, 5, 5), strides=(1, 2, 3), padding=(0, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 8, (1, 17, 21), strides=(1, 2, 2), padding=(0, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
                if input_action is not None:
                    conv3 = tf.layers.flatten(conv3)
                    conv3 = tf.concat([input_action, conv3], 1)
                    dense = tf.layers.dense(conv3, 1, activation=None, name = "dense/w")
                    return dense
            if self.net_type == 'STG_3_demo_2':
                """
                STG_action V0.3 20180215
                """
                conv1 = conv3d(inputs, 60, (3, 7, 7), strides=(1, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 60, (3, 25, 25), strides=(1, 2, 3), padding=(0, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 8, (1, 5, 10), strides=(1, 2, 2), padding=(0, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
                if input_action is not None:
                    conv3 = tf.layers.flatten(conv3)
                    conv3 = tf.concat([input_action, conv3], 1)
                    if dense_layer:
                        dense = tf.layers.dense(conv3, 50, activation=tf.nn.relu, name = "dense/w")
                        return dense
            elif self.net_type == 'STG_3_demo_1':
                """
                STG_action V0.2 20180214
                """
                conv1 = conv3d(inputs, 60, (3, 7, 7), strides=(1, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 60, (3, 25, 25), strides=(1, 2, 3), padding=(0, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 47, (1, 7, 15), strides=(1, 1, 1), padding=(0, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
                if input_action is not None:
                    conv3 = tf.layers.flatten(conv3)
                    conv3 = tf.concat([input_action, conv3], 1)
                    if dense_layer:
                        dense = tf.layers.dense(conv3, 50, activation=tf.nn.relu, name = "dense/w")
                        return dense
            elif self.net_type == 'STG_20180212':
                """
                STG_action V0.1 20180212
                """
                conv1 = conv3d(inputs, 120, (3, 7, 7), strides=(1, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (3, 25, 25), strides=(1, 2, 3), padding=(0, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 15, (1, 7, 15), strides=(1, 2, 2), padding=(0, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
                if input_action is not None:
                    conv3 = tf.layers.flatten(conv3)
                    conv3 = tf.concat([input_action, conv3], 1)
                    dense = tf.layers.dense(conv3, 50, activation=tf.nn.relu, name = "dense/w")
                    return dense
            elif self.net_type == 'STG_5':
                """
                This is for small frame
                """
                conv1 = conv3d(inputs, 120, (3, 7, 7), strides=(1, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (3, 30, 30), strides=(1, 2, 3), padding=(0, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 5, (1, 6, 9), strides=(1, 2, 2), padding=(0, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
            elif self.net_type == 'ST':
                """
                This is the spatial temporal model used for synthesizing dynamic textures with both spatial and temporal 
                stationarity. e.g. sea, ocean.
                """
                conv1 = conv3d(inputs, 120, (15, 15, 15), strides=(7, 7, 7), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 40, (7, 7, 7), strides=(3, 3, 3), padding="SAME", name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 20, (2, 3, 3), strides=(1, 2, 2), padding="SAME", name="conv3")
                conv3 = tf.nn.relu(conv3)
            elif self.net_type == 'FC_S':
                """
                This is the spatial fully connected model used for synthesizing dynamic textures with only temporal 
                stationarity with image size of 100. e.g. fire pot, flashing lights.
                """
                conv1 = conv3d(inputs, 120, (7, 7, 7), strides=(2, 2, 2), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (5, 50, 50), strides=(2, 2, 2), padding=(2, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 5, (2, 1, 1), strides=(1, 2, 2), padding=(1, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
            elif self.net_type == 'FC_S_large':
                """
                This is the spatial fully connected model for images with size of 224.
                """
                conv1 = conv3d(inputs, 120, (7, 7, 7), strides=(3, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (4, 75, 75), strides=(2, 1, 1), padding=(2, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 5, (2, 1, 1), strides=(1, 1, 1), padding=(1, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)
            else:
                return NotImplementedError

            return conv3
    def langevin_dynamics(self, samples, sample_a, gradient, gradient_a, batch_id,
                          update_state = True, update_action = True):
        for i in range(self.sample_steps):
            if update_state:
                noise = np.random.randn(*samples.shape)
                grad = self.sess.run(gradient, feed_dict = {self.syn: samples, self.syn_action: sample_a})
                samples = samples - 0.5 * self.step_size * self.step_size * (samples - grad) + self.step_size * noise
            if update_action:
                noise = np.random.randn(*sample_a.shape)
                grad_action = self.sess.run(gradient_a, feed_dict = {self.syn: samples, self.syn_action: sample_a})
                sample_a = sample_a - 0.5 * self.step_size * self.step_size * \
                    (sample_a - grad_action) + self.step_size * noise
            if self.pbar is not None:
                self.pbar.update(batch_id * self.sample_steps + i)
        return samples, sample_a

    def train(self, train_img, train_label):

        if np.max(train_img)>1:
            train_img = train_img / 255
        img_mean = train_img.mean()
        train_img = train_img - img_mean
        print(train_img.shape)
        self.image_size = list(train_img.shape[1:])

        self.obs = tf.placeholder(shape=[None] + self.image_size, dtype=tf.float32)
        self.syn = tf.placeholder(shape=[self.num_chain] + self.image_size, dtype=tf.float32)
        self.obs_action = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.syn_action = tf.placeholder(shape=[self.num_chain, self.action_size], dtype=tf.float32)

        obs_res = self.descriptor(self.obs, False, self.obs_action, self.dense_layer)
        syn_res = self.descriptor(self.syn, True, self.syn_action, self.dense_layer)
        train_loss = tf.subtract(tf.reduce_mean(syn_res,axis=0), tf.reduce_mean(obs_res,axis=0))

        train_loss_mean, train_loss_update = tf.contrib.metrics.streaming_mean(train_loss)

        recon_err_mean_1, recon_err_update_1 = tf.contrib.metrics.streaming_mean_squared_error(
            tf.reduce_mean(self.syn,axis=0),tf.reduce_mean(self.obs,axis=0))
        recon_err_mean_2, recon_err_update_2 = tf.contrib.metrics.streaming_mean_squared_error(
            tf.reduce_mean(self.syn_action, axis=0), tf.reduce_mean(self.obs_action, axis=0))
        recon_err_mean = recon_err_mean_1 + recon_err_mean_2
        recon_err_update = recon_err_update_1 + recon_err_update_2

        print('Network Established')
        dLdI = tf.gradients(syn_res, self.syn)[0]
        dLdI_action = tf.gradients(syn_res, self.syn_action)[0]


        num_batches = int(math.ceil(len(train_img) / self.batch_size))

        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in des_vars]
        reset_grads = [var.assign(tf.zeros_like(var)) for var in accum_vars]

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.beta1)
        grads_and_vars = optimizer.compute_gradients(train_loss, var_list=des_vars)
        update_grads = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads_and_vars)]
        des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in grads_and_vars if '/w' in var.name]
        # update by mean of gradients
        apply_grads = optimizer.apply_gradients([(tf.divide(accum_vars[i], num_batches), gv[1]) for i, gv in enumerate(grads_and_vars)])

        print('Initializing...')

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        sample_size = self.num_chain * num_batches
        # Modify from normal to uniform
        sample_video = np.random.random(size = [sample_size] + self.image_size)
        sample_action = np.random.random(size = [sample_size, self.action_size]) * 2 - 1

        tf.summary.scalar('train_loss', train_loss_mean)
        tf.summary.scalar('reconstruction_err', recon_err_mean)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        for epoch in range(self.num_epochs):

            gradients = []

            widgets = ["Epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            self.pbar = ProgressBar(maxval=num_batches * self.sample_steps, widgets=widgets)
            self.pbar.start()

            self.sess.run(reset_grads)
            for i in range(num_batches):

                obs_data = train_img[i * self.batch_size:min(len(train_img), (i+1) * self.batch_size)]
                obs_daction = train_label[i * self.batch_size:min(len(train_label), (i+1) * self.batch_size)]
                syn = sample_video[i * self.num_chain:(i+1) * self.num_chain]
                syn_action = sample_action[i * self.num_chain:(i+1) * self.num_chain]
                syn, syn_action = self.langevin_dynamics(syn, syn_action, dLdI, dLdI_action, i)
                grad = self.sess.run([des_grads, update_grads, train_loss_update],
                                     feed_dict={self.obs: obs_data, self.obs_action: obs_daction,
                                                self.syn: syn, self.syn_action: syn_action})[0]
                self.sess.run(recon_err_update, feed_dict={self.obs: obs_data, self.obs_action: obs_daction,
                                                           self.syn: syn, self.syn_action: syn_action})
                sample_video[i * self.num_chain:(i + 1) * self.num_chain] = syn
                sample_action[i * self.num_chain:(i + 1) * self.num_chain] = syn_action

                gradients.append(np.mean(grad))
            self.pbar.finish()

            self.sess.run(apply_grads)
            [loss, recon_err, summary] = self.sess.run([train_loss_mean, recon_err_mean, summary_op])
            print('Epoch #%d, descriptor loss: %.4f, SSD weight: %4.4f, Avg MSE: %4.4f' % (epoch, loss, float(np.mean(gradients)), recon_err))
            writer.add_summary(summary, epoch)

            if epoch % self.log_step == 0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleSequence(sample_video + img_mean, self.sample_dir, epoch, col_num=10)

                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

                saveSampleVideo(sample_video + img_mean, self.result_dir, global_step=epoch)

    def test(self, model_path, test_img, test_label):

        n_test = test_img.shape[0]
        assert (n_test == test_label.shape[0]), "Img and Label size mismatch."

        self.image_size = list(test_img.shape[1:])
        # Create some variables.
        self.obs = tf.placeholder(shape=[None] + self.image_size, dtype=tf.float32)
        self.syn = tf.placeholder(shape=[n_test] + self.image_size, dtype=tf.float32)
        self.obs_action = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.syn_action = tf.placeholder(shape=[n_test, self.action_size], dtype=tf.float32)

        obs_res = self.descriptor(self.obs, False, self.obs_action, self.dense_layer)
        syn_res = self.descriptor(self.syn, True, self.syn_action, self.dense_layer)

        print('Network Established')
        dLdI = tf.gradients(syn_res, self.syn)[0]
        dLdI_action = tf.gradients(syn_res, self.syn_action)[0]
        sample_action = np.random.random(size = [n_test, self.action_size]) * 2 - 1
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.import_meta_graph(model_path + '.meta')
        self.pbar = None

        with tf.Session() as sess:
            saver.restore(sess, model_path)
            print("Model loaded, start testing...")

            test_img, predicted_action = self.langevin_dynamics(test_img, sample_action, dLdI, dLdI_action, -1, False, True)
            score = evaluate_direct(predicted_action, test_label)
            print(score)
            return score, predicted_action

