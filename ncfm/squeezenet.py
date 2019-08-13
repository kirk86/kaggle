from keras.layers import Input, merge, GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Dense
from keras.reguralizers import l2
from keras.models import Model
from keras import backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
# from hyperopt.mongoexp import MongoTrials
from sklearn.metrics import log_loss
import numpy as np
from process import zca_whitening_matrix
from load import process_train, process_validation, process_test


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
nb_classes = 8


# function for fire node
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, 1, 1, border_mode='valid',
                      activation='relu', name=s_id + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid',
                         activation='relu', name=s_id + exp1x1)(x)

    right = Convolution2D(expand, 3, 3, border_mode='same',
                          activation='relu', name=s_id + exp3x3)(x)

    # SqueezeNet-Residual
    x = merge([left, right, x], mode='concat', concat_axis=channel_axis,
              name=s_id + 'concat')
    x = BatchNormalization(axis=channel_axis)(x)

    return x


# SqueezeNet architecture
def SqueezeNet(params, include_top=True, weights=None,
               input_tensor=None, input_shape=None):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid',
                      activation='relu', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    fire_id = 2
    squeeze = [16, 32]
    expand = [64, 128]
    pool_id = [3, 5]
    # 2 * fire_module + maxpool
    for num in range(2):
        x = fire_module(x, fire_id=fire_id, squeeze=params['squeeze1'],
                        expand=params['expand1'])

        x = Dropout(params['dropout1'])(x)
        x = fire_module(x, fire_id=fire_id + 1, squeeze=params['squeeze1'],
                        expand=params['expand1'])

        # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
        #                  name='pool' + str(pool_id[num]))(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Dropout(params['dropout2'])(x)

        fire_id += 2

    squeeze = [48, 48, 64, 64]
    expand = [192, 192, 256, 256]
    # 4 * fire_module
    for num in range(4):
        x = fire_module(x, fire_id=fire_id, squeeze=squeeze[num],
                        expand=expand[num])

        fire_id += 1

    # housekepping unecessary variables
    del fire_id, pool_id, squeeze, expand

    x = Dropout(params['dropout3'])(x)

    if include_top:
        x = Convolution2D(nb_classes, 1, 1, border_mode='valid',
                          activation='relu', name='conv10')(x)
        x = GlobalAveragePooling2D()(x)
        # implementation of SVM loss in the last layer
        # x = Dense(nb_classes, W_reguralizer=l2(0.01))(x)
        # x = Activation('linear')
        x = Activation('softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x, name='squeezenet')

    return model


# count = 0
# best = 0


def train_and_predict(params):
    # global best, count
    # count += 1
    model = SqueezeNet(weights=None, input_shape=(224, 224, 3))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    # this is for the SVM output
    # model.compile(loss='hinge',
    #               optimizer='adadelta', metrics=['accuracy'])

    model.fit(trX, trY, batch_size=16, nb_epoch=10, shuffle=True,
              validation_data=(valX, valY), verbose=1)

    scores = model.evaluate(valX, valY, verbose=0)
    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])

    y_val_pred = model.predict(valX)
    logLoss = log_loss(valY, y_val_pred)
    print("Log loss on valid set: => ", logLoss)

    # if logLoss < best:
    #     print 'new best loss:', logLoss, 'using', params
    #     best = logLoss
    # if count % 50 == 0:
    #     print 'iters:', count, ', loss:', logLoss, 'using', params

    # y_pred = model.predict(teX)
    # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # test_scores = model.evaluate(teX, y_pred, verbose=0)
    # print('Test loss:', test_scores[0])
    # print('Test accuracy:', test_scores[1])
    return {'loss': scores[0], 'status': STATUS_OK}


def objective(trials):
    space = {
        'dropout1': hp.uniform('dropout1', 0, 1),
        'dropout2': hp.uniform('dropout2', 0, 1),
        'dropout3': hp.uniform('dropout3', 0, 1),
        'squeeze1': hp.choice('squeeze1', [2, 4, 8, 16, 32, 64, 128]),
        'expand1': hp.choice('expand1', [32, 64, 128, 256, 512, 1024, 2048])
    }

    best = fmin(train_and_predict,
                space, algo=tpe.suggest, trials=trials, max_evals=100)
    print(best)
    print(space_eval(space, best))

np.random.seed(2018)

if __name__ == '__main__':
    valX, valY, valX_id = process_validation((224, 224), remove_mean=True)
    trX, trY, trX_id = process_train((224, 224), remove_mean=True)
    teX, teX_id = process_test((224, 224), remove_mean=True)

    # zca whitening
    names = ['valX', 'trX', 'teX']
    for dataset in range(3):
        for channel in range(3):
            tmp = eval(names[dataset])[:, :, :, channel].reshape(
                eval(names[dataset]).shape[0],
                np.prod(eval(names[dataset]).shape[1:3])
            )

            ZCA_valX = zca_whitening_matrix(tmp)

            eval(names[dataset])[:, :, :, channel] = np.dot(
                ZCA_valX,
                tmp
            ).reshape(tmp.shape[0], 224, -1)

    trials = Trials()
    objective(trials)
