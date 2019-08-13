import os
# import cv2
import glob
import shutil
import time
import json
# import subprocess
import numpy as np
# import pandas as pd
from PIL import Image
from keras.utils import np_utils
# from scipy.misc import imresize
from process import mean_removal
from matplotlib import pyplot as plt
from numba import jit
# import multiprocessing as mp

nb_classes = 8
data_dir = './data'
train_dir = 'train'
valid_dir = 'valid'
# test_dir = 'test/test_stg1'
test_dir = 'test'
bbox_dir = 'wei_bbox'


def read_img(path, shape, remove_mean=False):
    # img = cv2.imread(path)
    # return cv2.resize(img, shape, cv2.INTER_LINEAR)
    img = Image.open(path)
    img = img.convert('RGB')
    img_size = img.size
    img = img.resize(shape, Image.ANTIALIAS)
    if remove_mean:
        img = mean_removal(np.asarray(img, dtype='float32'))
    else:
        img = np.asarray(img)
    return img, img_size


# go through each image in each folder and read it
@jit
def load_train(shape, remove_mean=False):
    trX = []
    trX_id = []
    trY = []
    trX_img_sizes = dict()
    start = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for folder in folders:
        index = folders.index(folder)
        print('Load folder {} (Index: {})'.format(folder, index))
        path = os.path.join(data_dir, train_dir, folder, '*.jpg')
        files = glob.glob(path)
        for file_n in files:
            filename = os.path.basename(file_n)
            img, img_size = read_img(file_n, shape, remove_mean)
            trX.append(img)
            trX_id.append(filename)
            trY.append(index)
            trX_img_sizes[filename] = img_size

    end = time.time()
    print('Read train data time: {} seconds'
          .format(round(end - start, 2)))

    return trX, trY, trX_id, trX_img_sizes


# check if validation directory exists.
# if yes then create folders such in train.
# sample randomly 500 imgs from train
@jit
def load_validation(shape, remove_mean=False):
    valX = []
    valX_id = []
    valY = []
    valX_img_sizes = dict()
    start = time.time()

    if not os.path.exists(os.path.join(data_dir, valid_dir)) \
       and not os.path.isdir(os.path.join(data_dir, valid_dir)):
        print("Creating validation images...")
        create_validation()

    print("Read validation images")
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for folder in folders:
        index = folders.index(folder)
        print("Load folder {} (Index: {})".format(folder, index))
        path = os.path.join(data_dir, valid_dir, folder, '*.jpg')
        files = glob.glob(path)
        for file_n in files:
            filename = os.path.basename(file_n)
            img, img_size = read_img(file_n, shape, remove_mean)
            valX.append(img)
            valX_id.append(filename)
            valY.append(index)
            valX_img_sizes[filename] = img_size

    end = time.time()
    print("Read validation data time: {} seconds"
          .format(round(end - start, 2)))

    return valX, valY, valX_id, valX_img_sizes


# go through each image in the test folder and read it
@jit
def load_test(shape, remove_mean=False):
    path = os.path.join(data_dir, test_dir, '*.jpg')
    files = sorted(glob.glob(path))

    teX = []
    teX_id = []
    # teX_img_sizes = []
    for file_n in files:
        file_name = os.path.basename(file_n)
        img, _ = read_img(file_n, shape, remove_mean)
        teX.append(img)
        teX_id.append(file_name)

    return teX, teX_id


# create the proper validation directories if
# they don't exist
@jit
def create_validation():
    if not os.path.exists(os.path.join(data_dir, valid_dir)):
        os.mkdir(os.path.join(data_dir, valid_dir))
    else:
        # os.rmdir(os.path.join(data_dir, valid_dir))
        shutil.rmtree(os.path.join(data_dir, valid_dir))
        os.mkdir(os.path.join(data_dir, valid_dir))

    for directory in glob.glob(os.path.join(data_dir, train_dir, '*')):
        os.mkdir(os.path.join(data_dir,
                              valid_dir,
                              os.path.basename(directory)))

    train_imgs = glob.glob(os.path.join(data_dir, train_dir, '*/*.jpg'))
    # randomly permute the train images
    shuffled = np.random.permutation(train_imgs)
    for img in range(500):
        os.rename(shuffled[img],
                  os.path.join(data_dir,
                               valid_dir,
                               '/'.join(shuffled[img].split("/")[3:])))


# just transform data into numpy array
def process_train(shape, remove_mean=False):
    trX, trY, trX_id, trX_img_sizes = load_train(shape, remove_mean)
    # print len(trX), len(trY), len(trX_id)
    # print type(trX), type(trY), type(trX_id)

    print('Convert to  numpy...')
    trX = np.array(trX, dtype=np.uint8)
    trY = np.array(trY, dtype=np.uint8)

    print('Convert to float...')
    trX = trX.astype('float32')
    # trX /= 255
    trY = np_utils.to_categorical(trY, nb_classes)

    print('Train shape: ', trX.shape)
    print(trX.shape[0], 'train samples')

    return trX, trY, trX_id, trX_img_sizes


# transform validation data to numpy arrays
def process_validation(shape, remove_mean=False):
    valX, valY, valX_id, valX_img_sizes = load_validation(shape,
                                                          remove_mean)

    print('Convert to numpy...')
    valX = np.array(valX, dtype=np.uint8)
    valY = np.array(valY, dtype=np.uint8)

    print('Convert to float...')
    valX = valX.astype('float32')
    # valX /= 255
    valY = np_utils.to_categorical(valY, nb_classes)

    print('Validation shape: ', valX.shape)
    print(valX.shape[0], 'validation samples')

    return valX, valY, valX_id, valX_img_sizes


# transform test data into numpy array
def process_test(shape, remove_mean=False):
    start = time.time()
    teX, teX_id = load_test(shape, remove_mean)

    teX = np.array(teX, dtype=np.uint8)
    # test_data = test_data.transpose((0, 3, 1, 2))

    teX = teX.astype('float32')
    # teX /= 255

    print('Test shape: ', teX.shape)
    print(teX.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'
          .format(round(time.time() - start, 2)))

    return teX, teX_id


# todo: add emtpy bboxes for whenever they don't exist.
# rescale the bboxes based on image true sizes so that no problems
# arise when your rescale the imgs to feed to network
# for every image name get easy the corresponding bbox
# def load_bbox():
#     path = os.path.join(data_dir, bbox_dir, '*.json')
#     counter = 0
#     # print("Read and process bounding box labels.")
#     for bbox_file in glob.glob(path):
#         annot = json.loads(open(bbox_file).read())
#         for key, entry in zip(range(len(annot)), annot):
#             if 'annotations' in annot.keys() and len(entry['annotations']) >= 1:
#                 bb = {'x': entry['annotations'][0]['x'],
#                       'y': entry['annotations'][0]['y'],
#                       'width': entry['annotations'][0]['width'],
#                       'height': entry['annotations'][0]['height']}

#                 bb_new = rescale_bbox(bb, img_size, rescaled_img_size)

#                 bbox_final = {
#                             'image': entry['filename'].split('/')[-1],
#                             'class': entry['annotations'][0]['class'],
#                             'x': bb_new['x'],
#                             'y': bb_new['y'],
#                             'width': bb_new['width'],
#                             'height': bb_new['height']
#                         }

#                     # print("Annotation entry = ", counter, "has bbox")
#             else:  # construct empty bbox for missing bboxes
#                     bbox_final = {
#                             'image': entry['filename'].split('/')[-1],
#                             'class': entry['filename'].split('/')[0],
#                             'x': 0,
#                             'y': 0,
#                             'width': 0,
#                             'height': 0
#                         }
#                     print("entry = ", key, "doesn't have bbox in file ",
#                           path.split('/')[-1])
#             counter += 1

#     return bbox_final

@jit
def bboxes(raw_filenames):
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
    path = os.path.join(data_dir, bbox_dir)
    bb_json = {}
    for cls in classes:
        js = json.load(open('{}/{}.json'.format(path, cls), 'r'))
        for lbl in js:
            if 'annotations' in lbl.keys() and len(lbl['annotations']) > 0:
                # bb_json[lbl['filename'].split('/')[-1]] = sorted(
                #     lbl['annotations'],
                #     key=lambda x: x['height'] * x['width'])[-1]
                bb_json[lbl['filename'].split('/')[-1]] = lbl['annotations']

    file2idx = {o: i for i, o in enumerate(raw_filenames)}

    # empty_bbox = {'x': 0., 'y': 0., 'width': 0., 'height': 0.}

    # for f in raw_filenames:
    #     if f not in bb_json.keys():
    #         continue
    #         bb_json[f] = empty_bbox

    return bb_json


def convert_bb(bb, size, img_resized_shape):
    bb_params = ['x', 'y', 'width', 'height']
    bb = [bb[p] for p in bb_params]
    conv_x = (float(img_resized_shape) / size[0])
    conv_y = (float(img_resized_shape) / size[1])
    bb[0] = bb[0] * conv_x
    bb[1] = bb[1] * conv_y
    bb[2] = max(bb[2] * conv_x, 0)
    bb[3] = max(bb[3] * conv_y, 0)

    return bb


def convert_bb_alt(bb, size, img_resized_shape):
    bb_params = ['height', 'width', 'x', 'y']
    bb = [bb[p] for p in bb_params]
    bb[0] = bb[0] / size[1]
    bb[1] = bb[1] / size[0]
    bb[2] = bb[2] / size[0]
    bb[3] = bb[3] / size[1]
    return bb


def rescale_bbox(bbox, true_img_size, rescaled_img):
    params = ['x', 'y', 'width', 'height']
    bbox = [bbox[p] for p in params]
    x_ratio = (float(rescaled_img) / true_img_size[0])
    y_ratio = (float(rescaled_img) / true_img_size[1])
    bbox[0] = bbox[0] * x_ratio
    bbox[1] = bbox[1] * y_ratio
    bbox[2] = max(bbox[2] * x_ratio, 0)
    bbox[3] = max(bbox[3] * y_ratio, 0)

    return bbox


@jit
def complete_bbox(raw_filenames, sizes, img_resize_shape):
    bbxs = bboxes(raw_filenames)
    for f in bbxs.keys():
        for key, bbox in enumerate(bbxs[f]):
            bbxs[f][key] = rescale_bbox(bbox, sizes[f], img_resize_shape)

    # trX_bbox = np.stack([rescale_bbox(bbox, sizes[f], img_resize_shape)
    #                      for f in bbxs.keys()
    #                      for key, bbox in enumerate(bbxs[f])]
    #                     ).astype(np.float32)
    # trX_bbox = np.stack([convert_bb(bbxs[f], sizes[f], img_resize_shape)
    #                      for f in bbxs.keys()]
    #                     ).astype(np.float32)

    return bbxs


def create_rect(bb, color='red'):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3],
                         color=color, fill=False, lw=3)


def show_bb(bb):
    # bb = val_bbox[i]
    # plot(val[i])
    plt.gca().add_patch(create_rect(bb))




# missing annotations from label ROIs

# DOL = 12 missing
# [img_06773.jpg, img_05444.jpg]

# ALB = 8 missing

# [u'img_07008.jpg',
#  u'img_06460.jpg',
#  u'img_04798.jpg',
#  u'img_02292.jpg',
#  u'img_01958.jpg',
#  u'img_00576.jpg',
#  u'img_00568.jpg',
#  u'img_00425.jpg']

# BET = 1 missing
# [u'img_00379.jpg']

# LAG = 0 missing

# OTHER = 0 missing

# SHARK = 1 missing
# [u'../data/train/SHARK/img_06082.jpg']

# YFT = 3 missing

# [u'../data/train/YFT/img_04558.jpg',
#  u'../data/train/YFT/img_03183.jpg',
#  u'../data/train/YFT/img_02785.jpg']
