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
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-num_chain', type=int, default=30, help='number of synthesized results for each batch of training data')
    parser.add_argument('-num_frames', type=int, default=5, help='number of frames used in training data')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-beta1', type=float, default=0.5, help='momentum1 in Adam')
    parser.add_argument('-net_type', type=str, default='STG_5_V2.0', help='Net type')
    parser.add_argument('-dense_layer', type=float, default=1, help='Net type')
    parser.add_argument('-action_cold_start', type=int, default=1, help='Whether use cold start on action')
    parser.add_argument('-state_cold_start', type=int, default=1, help='Whether use cold start on state')

    # langevin hyper-parameters
    parser.add_argument('-delta', '--step_size', type=float, default=0.3)
    parser.add_argument('-action_delta', '--action_step_size', type=float, default=0.3)
    parser.add_argument('-sample_steps', type=int, default=30)
    parser.add_argument('-action_sample_steps', type=int, default=1)

    # misc
    parser.add_argument('-output_dir', type=str, default='./output', help='output directory')
    parser.add_argument('-category', type=str, default='V2.0_coldstart')
    parser.add_argument('-data_path', type=str, default='./training_demo', help='root directory of data')
    parser.add_argument('-log_step', type=int, default=30, help='number of steps to output synthesized image')
    
    parser.add_argument('-model_path', type=str, default='V1.4-3_warmstart_delta0.001', help='root directory of data')

    opt = parser.parse_args()
    opt.model_path = 'output/' + opt.category + '/model/model.ckpt-480'

    # Prepare training data
    train_img, train_label = loadActionDemo(opt.data_path, 100)
    # Split 8000 Frame into multiple small snap
    num_gif = 100
    resize_size = [55,100]
    train_label, train_img = SplitFrame(train_label, train_img, resize_size, opt.num_frames, num_gif)
    train_label[:, 0] = (train_label[:, 0] + 0.3) / 0.6 * 255
    train_label[:, 1] = train_label[:, 1] / 0.6 * 255
    train_label[:, 2] = train_label[:, 2] / 0.5 * 255
    mp.plot(train_label[:,0], 'r')
    mp.plot(train_label[:, 1], 'b')
    mp.plot(train_label[:, 2], 'g')
    mp.savefig('true_action.png')
    with tf.Session() as sess:
        model = STGConvnet(sess, opt)
        score, energy, predicted_action = model.test(opt.model_path, train_img, train_label)

    data = {'score': score, 'energy' : energy, 'predicted_action' : predicted_action, 'train_label' : train_label}
    np.save('test', data)
    mp.clf()
    mp.plot(predicted_action[:,0], 'r')
    mp.plot(predicted_action[:, 1], 'b')
    mp.plot(predicted_action[:, 2], 'g')
    mp.savefig('predicted_action.png')

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

