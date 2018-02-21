import argparse
import tensorflow as tf
from src.model import STGConvnet
from src.util import *
import numpy as np
from src.ops import *
import matplotlib.pyplot as mp

def main():
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-num_chain', type=int, default=30, help='number of synthesized results for each batch of training data')
    parser.add_argument('-num_frames', type=int, default=3, help='number of frames used in training data')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-beta1', type=float, default=0.5, help='momentum1 in Adam')
    parser.add_argument('-net_type', type=str, default='STG_3_demo_4', help='Net type')
    parser.add_argument('-dense_layer', type=float, default=1, help='Net type')
    parser.add_argument('-action_cold_start', type=int, default=1, help='Net type')


    # langevin hyper-parameters
    parser.add_argument('-delta', '--step_size', type=float, default=0.3)
    parser.add_argument('-sample_steps', type=int, default=30)

    # misc
    parser.add_argument('-output_dir', type=str, default='./output', help='output directory')
    parser.add_argument('-category', type=str, default='demo_0')
    parser.add_argument('-data_path', type=str, default='./training_demo', help='root directory of data')
    parser.add_argument('-log_step', type=int, default=20, help='number of steps to output synthesized image')
    
    parser.add_argument('-model_path', type=str, default='output/demo_4_coldstart_withscale/model/model.ckpt-495', help='root directory of data')

    opt = parser.parse_args()

    # Prepare training data
    train_img, train_label = loadActionDemo(opt.data_path, 100)
    # Split 8000 Frame into multiple small snap
    num_gif = 100
    train_label, train_img = SplitFrame(train_label, train_img, opt.num_frames, num_gif)

    with tf.Session() as sess:
        model = STGConvnet(sess, opt)
        score, energy, predicted_action = model.test(opt.model_path, train_img, train_label)

    mp.plot(predicted_action[:,0])
    mp.show()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # with tf.Session() as sess:
    #     # Restore variables from disk.
    #     saver.restore(sess, import_path + "model.ckpt-480")
    #     print("Model loaded, start testing...")

        # saver = tf.train.import_meta_graph(import_path+"model.ckpt-480.meta")
        # with tf.Session() as sess:
        #     .data-00000-of-00001")
        #
        #

def desc(self):
    with tf.variable_scope('test'):
        w = tf.get_variable('w', [10], initializer=tf.random_normal_initializer())

if __name__ == '__main__':
    main()

