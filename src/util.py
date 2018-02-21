from __future__ import division


import os
import numpy as np
import math
from PIL import Image
import scipy.misc
import subprocess

def loadDemo(data_path, resize_size):
    # Read human_demo.txt
    txt_name = [ss for ss in os.listdir(data_path) if ss.endswith(".txt")]
    assert len(txt_name) > 0, 'Error | Loading data: No label found!'
    demo_label_file = open(os.path.join(data_path, txt_name[0]), 'r')
    labels = demo_label_file.readlines()
    label_size = len(labels[0].split())

    screenshots = sorted([ss for ss in os.listdir(data_path) if ss.endswith(".png")])
    num_demo = len(screenshots)
    assert num_demo == len(labels), 'Error | Loading data: Number of label error!'

    data_image = np.empty((num_demo, resize_size[0], resize_size[1], 3))
    data_label = np.empty((num_demo, label_size))
    for i, ss in enumerate(screenshots):
        img = scipy.misc.imread(os.path.join(data_path,ss), mode='RGB')
        data_image[i, ...] = scipy.misc.imresize(img, size=resize_size)
        data_label[i, ...] = [float(e) for e in labels[i].replace('\n', '').split()]

    return data_image, data_label

def loadActionDemo(data_path, cut = -1):
    data = np.load(os.path.join(data_path,'demo.npz'))
    images = data['imgs']
    actions = data['actions']
    if cut > 0:
        images = images[:cut, ...]
        actions = actions[:cut, ...]
    return images, actions

def SplitFrame(data_label, data_image, resize_size = None, num_frame = 5, split_at = 0):

    if resize_size is None:
        resize_size = data_image.shape[1:2]
    num_demo = data_image.shape[0]
    img_temp = np.empty([num_demo] + resize_size + [data_image.shape[3]])
    for i in range(num_demo):
        img_temp[i, ...] = scipy.misc.imresize(data_image[i, ...], size=resize_size)

    if split_at == 0:
        num_gif = num_demo - num_frame + 1
        nf = np.arange(num_gif)
    else:
        num_gif_ori = num_demo // split_at
        nf = np.concatenate([np.arange(i*split_at, (i+1)*split_at - num_frame + 1) for i in range(num_gif_ori)])
        num_gif = len(nf)
    shape = [num_gif, num_frame] + list(img_temp.shape[1:])
    data_output = np.empty(shape)
    for i,j in enumerate(nf):
        data_output[i,...] = img_temp[j:(j+num_frame), ...]
    return data_label[nf + num_frame - 1,...], data_output

def evaluate_direct(predicted_label, truth_label):

    score = ((predicted_label - truth_label)**2).sum()
    return score

# Inherited from STGConvnet
def loadVideoToFrames(data_path, syn_path, ffmpeg_loglevel = 'quiet'):
    videos = [f for f in os.listdir(data_path) if f.endswith(".avi") or f.endswith(".mp4")]
    num_videos = len(videos)

    for i in range(num_videos):
        video_path = os.path.join(data_path, videos[i])
        out_dir = os.path.join(syn_path, "sequence_%d" % i)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            subprocess.call('ffmpeg -loglevel {} -i {} {}/%03d.png'.format(ffmpeg_loglevel,video_path, out_dir), shell=True)
    return num_videos

# Inherited from STGConvnet
def cell2img(filename, out_dir='./final_result',image_size=224, margin=2):
    img = scipy.misc.imread(filename, mode='RGB')
    num_cols = img.shape[1] // image_size
    num_rows = img.shape[0] // image_size
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = img[ir*(image_size+margin):image_size + ir*(image_size+margin),
                   ic*(image_size+margin):image_size + ic*(image_size+margin),:]
            scipy.misc.imsave("%s/%03d.png" % (out_dir,ir*num_cols+ic), temp)
    print(img.shape)

# Inherited from STGConvnet
def img2cell(images, col_num=10, margin=2):
    [num_images, size_h, size_w, num_channel] = images.shape
    row_num = int(math.ceil(num_images/col_num))
    saved_img = np.zeros(((row_num * size_h + margin * (row_num - 1)),
                          (col_num * size_w + margin * (col_num - 1)),
                          num_channel), dtype=np.float32)
    for idx in range(num_images):
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = np.squeeze(images[idx])
        temp = np.maximum(0.0, np.minimum(255.0, np.round(temp)))
        gLow = temp.min()
        gHigh = temp.max()
        temp = (temp - gLow) / (gHigh - gLow)
        saved_img[(size_h + margin) * ir:size_h + (size_h + margin) * ir,
        (size_w + margin) * ic:size_w + (size_w + margin) * ic, :] = temp
    return saved_img

# Inherited from STGConvnet
def getTrainingData(data_path, num_frames=70, image_size=100, isColor=True, postfix='.png'):
    num_channel = 3
    if not isColor:
        num_channel = 1
    videos = [f for f in os.listdir(data_path) if f.startswith('sequence')]
    num_videos = len(videos)
    images = np.zeros(shape=(num_videos, num_frames, image_size, image_size, num_channel))
    for iv in range(num_videos):
        video_path = os.path.join(data_path, 'sequence_%d' % iv)
        imgList = [f for f in os.listdir(video_path) if f.endswith(postfix)]
        imgList.sort()
        imgList = imgList[:num_frames]
        for iI in range(len(imgList)):
            image = Image.open(os.path.join(video_path, imgList[iI])).resize((image_size, image_size), Image.BILINEAR)
            if isColor:
                image = np.asarray(image.convert('RGB')).astype(float)
            else:
                image = np.asarray(image.convert('L')).astype(float)
                image = image[..., np.newaxis]
            images[iv, iI, :,:,:] = image
    return images.astype(float)

# Inherited from STGConvnet
def saveSampleVideo(samples, out_dir, global_step=None, ffmpeg_loglevel='quiet', fps=25):
    [num_video, num_frames, image_size, _, _] = samples.shape

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for ifr in range(num_frames):
        saved_img = img2cell(np.squeeze(samples[:, ifr, :, :, :]))
        scipy.misc.imsave("%s/step_%04d_%03d.png" % (out_dir, global_step, ifr), saved_img)

# Inherited from STGConvnet
def saveSampleSequence(samples, sample_dir, iter, col_num=10):
    num_video  = samples.shape[0]

    for iv in range(num_video):
        save_dir = os.path.join(sample_dir, "sequence_%d" % iv)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        scipy.misc.imsave("%s/%04d.png" % (save_dir, iter), img2cell(samples[iv], col_num=col_num))