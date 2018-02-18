import argparse
import tensorflow as tf
from src.model import STGConvnet
from src.util import *

def main():
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-num_chain', type=int, default=30, help='number of synthesized results for each batch of training data')
    parser.add_argument('-num_frames', type=int, default=3, help='number of frames used in training data')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-beta1', type=float, default=0.5, help='momentum1 in Adam')
    parser.add_argument('-net_type', type=str, default='STG_3', help='Net type')
    parser.add_argument('-dense_layer', type=float, default=1, help='Net type')


    # langevin hyper-parameters
    parser.add_argument('-delta', '--step_size', type=float, default=0.3)
    parser.add_argument('-sample_steps', type=int, default=20)

    # misc
    parser.add_argument('-output_dir', type=str, default='./output', help='output directory')
    parser.add_argument('-category', type=str, default='demo_0')
    parser.add_argument('-data_path', type=str,
                        default='./training_demo', help='root directory of data')
    parser.add_argument('-log_step', type=int, default=20, help='number of steps to output synthesized image')

    opt = parser.parse_args()

    # Prepare training data
    train_img, train_label = loadActionDemo(opt.data_path, 800)
    # Split 8000 Frame into multiple small snap
    num_gif = 100
    train_label, train_img = SplitFrame(train_label, train_img, opt.num_frames, num_gif)

    with tf.Session() as sess:
        model = STGConvnet(sess, opt)
        model.train(train_img, train_label)

if __name__ == '__main__':
    main()