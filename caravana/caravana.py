# coding: utf-8

import numpy as np
# import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate
from keras.layers import BatchNormalization, AlphaDropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
# from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import os
import cv2
import time
from keras.preprocessing.image import load_img

# set the necessary directories
data_dir = "./data/train/"
mask_dir = "./data/train_masks/"
contour_dir = "./data/contours/"
test_dir = "./data/final_test/test/"
all_images = os.listdir(data_dir)
# %ls data


# In[3]:


# pick which images we will use for testing and which for validation
train_images, validation_images = train_test_split(
    all_images,
    train_size=0.8,
    test_size=0.2
)
# test_images = glob.glob(test_dir + "*.jpg")
test_images = os.listdir(test_dir)
# print(len(train_images), len(validation_images), len(test_images))


def preprocess(data_dir, dims, rles, img_name):
    if rles:
        img = cv2.resize(img_name, (1918, 1280))
        mask = img > 0.5
        img = rle(mask)
    else:
        img = load_img(data_dir+img_name)
        img = np.array(img, dtype='float32')/255.
#       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dims)
    return img


# utility function to convert greyscale images to rgb
def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img


# generator that we will use to read the data from the directory
def process_data(data_dir, mask_dir, contour_dir, batch_size, dims, images):
    """
    data_dir: where the actual images are kept
    mask_dir: where the actual masks are kept
    images: the filenames of the images we want to generate batches from
    batch_size: self explanatory
    dims: the dimensions in which we want to rescale our images
    """
#     while True:
    imgs = []
    labels = []
    contours = []
    # images
    img = preprocess(data_dir, dims, False, images)
    imgs.append(img)

    # masks
    mask = preprocess(
        mask_dir,
        dims, False,
        images.split(".")[0] + '_mask.gif'
    )
#   mask = load_img(mask_dir+train_images[i].split(".")[1] + '_mask.gif')
    labels.append(mask[:, :, 0])

    contour = preprocess(
        contour_dir,
        dims, False,
        images.split(".")[0] + '_mask.png'
    )
    contours.append(contour[:, :, 0])

    return imgs, contours, labels


def generator(images, batch_size=len(all_images)):
    from functools import partial
    ix = np.random.choice(
        np.arange(len(images)),
        batch_size
    )  # from len(train_images) choose batch_size=64
    pool = mp.Pool(processes=cpu_count())
    train_gen = partial(
        process_data,
        data_dir,
        mask_dir,
        contour_dir,
        batch_size,
        (256, 256)
    )
    gen = pool.map_async(train_gen, list(np.array(images)[ix]), chunksize=8)
    gen.wait()
    results = gen.get()
    pool.close()
    pool.join()
    pool.terminate()
    x, contour, mask = zip(*results)
    x = np.array(x, dtype='float32').reshape(-1, 256, 256, 3)
    contour = np.array(contour, dtype='int32').reshape(-1, 256, 256, 1)
    mask = np.array(mask, dtype='int32').reshape(-1, 256, 256, 1)

    return x, contour, mask

# example use
# train_gen = data_gen_small(data_dir, mask_dir, train_images, 64, (256, 256))
# val_gen = data_gen_small(data_dir, mask_dir, validation_images, 64, (256, 256))
# img, msk = next(train_gen)
# val_img, val_msk = next(val_gen)
# fig = plt.figure(figsize=(10, 10))
# plt.subplot(121)
# plt.imshow(img[0])
# plt.imshow(grey2rgb(msk[0]), alpha=0.5)
# plt.subplot(122)
# plt.imshow(val_img[0])
# plt.imshow(grey2rgb(val_msk[0]), alpha=0.5)


# Now let's use Tensorflow to write our own dice_coeficcient metric
def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


# First let's define the two different types of layers that we will be using.
def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='elu')(input_layer)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='elu')(conv1)
    residual = BatchNormalization(axis=3)(conv2)
    if pool:
        max_pool = MaxPool2D()(residual)
#         max_pool = AveragePooling2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, (2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    # drop = Dropout(0.3)(concat)
    drop = AlphaDropout(rate=0.3)(concat)
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='elu')(drop)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='elu')(conv1)
    return conv2


def create_net():
    # Make a custom U-nets implementation.
    filters = 64
    input_layer = Input(shape=[256, 256, 3])
    layers = [input_layer]
    residuals = []

    # Down 1, 128
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)

    filters *= 2

    # Down 2, 64
    d2, res2 = down(d1, filters)
    residuals.append(res2)

    filters *= 2

    # Down 3, 32
    d3, res3 = down(d2, filters)
    residuals.append(res3)

    filters *= 2

    # Down 4, 16
    d4, res4 = down(d3, filters)
    residuals.append(res4)

    filters *= 2

    # Down 5, 8
    d5 = down(d4, filters, pool=False)

    # Up 1, 16
    up1 = up(d5, residual=residuals[-1], filters=filters/2)

    filters /= 2

    # Up 2,  32
    up2 = up(up1, residual=residuals[-2], filters=filters/2)

    filters /= 2

    # Up 3, 64
    up3 = up(up2, residual=residuals[-3], filters=filters/2)

    filters /= 2

    # Up 4, 128
    up4 = up(up3, residual=residuals[-4], filters=filters/2)

    contour = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)
    mask = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

    model = Model(input_layer, [contour, mask])
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['mse',  dice_coef]
    )
    model.summary()

    return model


def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten() == 1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b > prev + 1):
            runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b

    return ' '.join([str(i) for i in runs])


def weighted_dice_coef(y_true, y_pred):
    mean = 0.21649066
    w_1 = 1/mean**2
    w_0 = 1/(1-mean)**2
    y_true_f_1 = K.flatten(y_true)
    y_pred_f_1 = K.flatten(y_pred)
    y_true_f_0 = K.flatten(1-y_true)
    y_pred_f_0 = K.flatten(1-y_pred)

    intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)
    intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)

    return 2 * (w_0 * intersection_0 + w_1 * intersection_1) / ((w_0 * (K.sum(y_true_f_0) + K.sum(y_pred_f_0)))
                                                                + (w_1 * (K.sum(y_true_f_1) + K.sum(y_pred_f_1))))


def predict_masks(test_images, dims=(256, 256), batch_size=32):
    from functools import partial
#     valid_imgs = []
    rles = []
    tic = time.time()
    downscale = partial(preprocess, test_dir, dims, False)
    upscale = partial(preprocess, test_dir, dims, True)
    pool = mp.Pool(processes=cpu_count())
    for batch in xrange(0, len(test_images), batch_size):
#         resized = pool.map_async(
#         preprocess,
        # zip(repeat(test_dir), test_images[batch:batch+batch_size],
        #     repeat(dims), repeat(False))
        # )
        resized = pool.map_async(downscale, test_images[batch:batch+batch_size])
        resized.wait()
        contours, masks = model.predict_on_batch(np.array(resized.get()))
#         valid_imgs.append(np.squeeze(predictions))
#         masks = pool.map_async(preprocess, zip(repeat(test_dir), test_images[batch:batch+batch_size],
#                                                repeat(dims), repeat(True)))
        upscaled_masks = pool.map_async(upscale, masks)
        masks.wait()
        rles.append(upscaled_masks.get())

        print("{}:{}, {}, {}, {}".format(
            batch,
            batch+batch_size,
            len(test_images[batch:batch+batch_size]),
            np.array(resized.get()).shape,
            len(upscaled_masks.get())
            )
        )

    pool.close()
    pool.join()
    pool.terminate()
    print("{} min.".format((time.time() - tic)/60.))
    return rles


if __name__=="__main__":

    tic = time.time()
    x, contours, masks = generator(all_images)
    print((time.time() - tic)/60.)
    # model = create_net()
    # model.load_weights('weights.49-0.01.hdf5')
    model = load_model('unet-model-contours.hdf5', custom_objects={'dice_coef':dice_coef})
    print("Weights loaded into the model.")

    model.fit(x, [contours, masks], batch_size=86, epochs=10, validation_split=0.12, verbose=2,
              callbacks=[
                  ModelCheckpoint('weights-contours.{epoch:02d}-{val_loss:.2f}.hdf5',
                                  monitor='val_dice_coef',
                                  save_weights_only=True,
                                  verbose=0)
                  ]
              )
    model.save('unet-model-contours.hdf5')
