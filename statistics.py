
import numpy as np 
import scipy.misc
import matplotlib.pyplot as mp
import tensorflow as tf
from src.model import STGConvnet
from src.util import *

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

output_histogram()