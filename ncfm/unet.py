from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D
from keras.layers import UpSampling2D, Reshape, Activation, Dropout
from keras.layers import Deconvolution2D, Dense, Flatten, Input
from keras.layers import Permute
from keras.optimizers import Adam, SGD
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
# from keras.utils.visualize_util import plot

# Define the neural network gnet
# change function call "get_unet" to "get_gnet" line 166 before use


# def gnet(img_shape):
#     inputs = Input(shape=img_shape)
#     conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#     up1 = UpSampling2D(size=(2, 2))(conv1)
#     #
#     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
#     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     #
#     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
#     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     #
#     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
#     conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
#     #
#     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
#     conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)
#     #
#     up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
#     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up2)
#     conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
#     #
#     up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
#     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up3)
#     conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)
#     #
#     up4 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
#     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up4)
#     conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)
#     #
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
#     conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
#     conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
#     #
#     conv10 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv9)
#     conv10 = Reshape((2, img_shape[0] * img_shape[1]))(conv10)
#     conv10 = Permute((2, 1))(conv10)
#     A
#     ############
#     conv10 = Activation('softmax')(conv10)
#     model = Model(input=inputs, output=conv10)

#     # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
#     model.compile(optimizer='sgd', loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     return model


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = (2. * K.sum(y_true_f * y_pred_f) + smooth)
    normalization = (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return intersection / normalization


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    # return dice_coef(y_true, y_pred)


# example of deconvolution
# model = Sequential([
#     Convolution2D(32, 3, 3, activation='relu', border_mode='same',
#                   input_shape=(512, 512, 3)),
#     Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
#     Convolution2D(256, 3, 3, activation='relu', border_mode='same'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Deconvolution2D(128, 2, 2, output_shape=(None, 256, 256, 128),
#                     border_mode='same', subsample=(2, 2)),
#     Deconvolution2D(64, 2, 2, output_shape=(None, 512, 512, 64),
#                     border_mode='same', subsample=(2, 2)),
#     Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same'),
#     Reshape((512, 512))
# ])

# model2 = Sequential([
#     Convolution2D(32, 3, 3, activation='relu', border_mode='same',
#                   input_shape=(512, 512, 3)),
#     Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
#     Convolution2D(256, 3, 3, activation='relu', border_mode='same'),
#     MaxPooling2D(pool_size=(2, 2)),
#     UpSampling2D(),
#     UpSampling2D(),
#     Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same'),
#     Reshape((512, 512))
# ])

def unet(img_shape):
    inputs = Input(shape=img_shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu',
                          border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv_ext1 = Convolution2D(1024, 3, 3, activation='relu',
                              border_mode='same')(pool5)
    conv_ext1 = Convolution2D(1024, 3, 3, activation='relu',
                              border_mode='same')(conv_ext1)

    up_ext6 = merge([Deconvolution2D(512, 2, 2,
                                     output_shape=(None, 32, 32, 512),
                                     activation='relu', subsample=(2, 2),
                                     border_mode='same')(conv_ext1), conv5],
                    mode='concat', concat_axis=3)
    conv_ext2 = Convolution2D(512, 3, 3, activation='relu',
                              border_mode='same')(up_ext6)
    conv_ext2 = Convolution2D(512, 3, 3, activation='relu',
                              border_mode='same')(conv_ext2)

    up6 = merge([Deconvolution2D(256, 2, 2,
                                 output_shape=(None, 64, 64, 256),
                                 activation='relu', subsample=(2, 2),
                                 border_mode='same')(conv_ext2), conv4],
                mode='concat', concat_axis=3)
    # # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv_ext2],
    # #             mode='concat', concat_axis=3)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu',
                          border_mode='same')(conv6)

    up7 = merge([Deconvolution2D(128, 2, 2,
                                 output_shape=(None, 128, 128, 128),
                                 activation='relu', subsample=(2, 2),
                                 border_mode='same')(conv6), conv3],
                mode='concat', concat_axis=3)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu',
                          border_mode='same')(conv7)

    up8 = merge([Deconvolution2D(64, 2, 2,
                                 output_shape=(None, 256, 256, 64),
                                 activation='relu', subsample=(2, 2),
                                 border_mode='same')(conv7), conv2],
                mode='concat', concat_axis=3)
    conv8 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu',
                          border_mode='same')(conv8)

    up9 = merge([Deconvolution2D(32, 2, 2,
                                 output_shape=(None, 512, 512, 32),
                                 activation='relu', subsample=(2, 2),
                                 border_mode='same')(conv8), conv1],
                mode='concat', concat_axis=3)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu',
                          border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    flat = Flatten(name='flatten')(conv9)
    dense1 = Dense(64, activation='relu', name='fc1')(flat)
    dense2 = Dense(64, activation='relu', name='fc2')(dense1)
    dense = Dense(8, activation='softmax', name='predictions')(dense2)

    model = Model(input=inputs, output=[conv10, dense])
    # model = Model(input=inputs, output=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=[dice_coef_loss,
                                                 'categorical_crossentropy'],
                  metrics=[dice_coef, 'accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=[dice_coef_loss],
    #               metrics=[dice_coef])

    return model


# model = unet((512,512,3))
# model.fit(np.random.randn(10,512,512,3), np.random.randn(10,512,512,1),
# nb_epoch=3, batch_size=2, verbose=1)
