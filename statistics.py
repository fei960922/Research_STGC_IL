
import numpy as np 
import scipy.misc
import matplotlib.pyplot as mp
import tensorflow as tf
from src.model import STGConvnet
from src.util import *
from src.ops import *
from tensorflow.python.tools import inspect_checkpoint as chkp

def output_all(images):
   
    for i, img in enumerate(images):
        print(i)
        scipy.misc.imsave("All/%04d.png" % (i), img)
    images = data['imgs']
    action = data['actions']
    train_label = action.copy()
    if np.max(train_label) < 1:
        train_label[:, 0] = (train_label[:, 0] + 0.3) / 0.6 * 255
        train_label[:, 1] = train_label[:, 1] / 0.6 * 255
        train_label[:, 2] = train_label[:, 2] / 0.5 * 255
    mp.plot(action[:, 0])
    mp.show()
    mp.plot(train_label[:, 0])
    mp.show()
    mp.plot(action[:, 1])
    mp.show()
    mp.plot(train_label[:, 1])
    mp.show()
    mp.plot(action[:, 2])
    mp.show()
    mp.plot(train_label[:, 2])
    mp.show()

def output_histogram():
    train_img, train_label = loadActionDemo('./training_demo', 400)
    img_mean = train_img.mean()
    train_label[:, 0] = (train_label[:, 0] + 0.3) / 0.6
    train_label[:, 1] = train_label[:, 1] / 0.6
    train_label[:, 2] = train_label[:, 2] / 0.5

    #train_img = train_img - img_mean
    print(train_img.reshape(-1,1).shape)
    print('Showing...')
    mp.hist(train_label, bins=50)
    mp.savefig('output.png')

def test_restore():
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

    image_size = [5, 55, 100, 3]
    inc_v1 = v1.assign(v1 + 1)
    dec_v2 = v2.assign(v2 - 1)
    obs = tf.placeholder(shape=[None] + image_size, dtype=tf.float32)
    syn = tf.placeholder(shape=[12] + image_size, dtype=tf.float32)
    obs_action = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    syn_action = tf.placeholder(shape=[12, 3], dtype=tf.float32)

    conv1 = conv3d_leaky_relu(obs, 120, (3, 5, 5), strides=(1, 2, 3), padding="VALID", name="conv1")
    conv2 = conv3d_leaky_relu(conv1, 30, (3, 5, 5), strides=(1, 2, 2), padding=(0, 0, 0), name="conv2")
    conv3 = conv3d_leaky_relu(conv2, 25, (1, 11, 14), strides=(1, 2, 2), padding=(0, 0, 0), name="conv3")

    fc1 = tf.layers.dense(obs_action, 50, activation=tf.nn.leaky_relu, name="fc1/w")
    fc2 = tf.layers.dense(fc1, 25, activation=tf.nn.leaky_relu, name="fc2/w")
    concat_layer = tf.concat([fc2, tf.layers.flatten(conv3)], 1)
    fc3 = tf.layers.dense(concat_layer, 50, activation=tf.nn.leaky_relu, name="fc3/w")
    obs_res = tf.layers.dense(fc3, 1, activation=tf.nn.leaky_relu, name="fc4/w")

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.
        inc_v1.op.run()
        dec_v2.op.run()
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

    tf.reset_default_graph()
    model_path = "output/V2.0-all_cold_start/model/model.ckpt-480"
    #chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True, all_tensor_names='')
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])

    # Add ops to save and restore all the variables.
    saver = tf.train.import_meta_graph(model_path + '.meta')
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_path)
        print("Model restored.")
        # Check the values of the variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print("%s with value %s" % (v.name, sess.run(v)))

test_restore()