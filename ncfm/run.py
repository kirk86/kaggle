from load import process_train, process_validation, process_test
from load import complete_bbox
# from squeezenet import SqueezeNet, objective
from sklearn.metrics import log_loss
# from hyperopt import Trials
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import pandas as pd
# from unet import unet
# from hyperopt import Trials
from models import xception_model
# from process import remove_mean
from sklearn.feature_extraction.image import extract_patches_2d


np.random.seed(2017)


if __name__ == '__main__':
    # valX, valY, valX_id, valX_img_sizes = process_validation((512, 512))
    trX, trY, trX_id, trX_img_sizes = process_train((512, 512))
    teX, teX_id = process_test((512, 512))
    trX = trX / 255.
    teX = teX / 255.

    trX_bbx = complete_bbox(trX_id, trX_img_sizes, 512)
    import pdb
    pdb.set_trace()
    # valX_bbx = complete_bbox(valX_id, valX_img_sizes, 512)

    # valX_mask = np.zeros_like(valX)
    # trX_mask = np.zeros_like(trX)

    # for idx, bbx, img in zip(xrange(len(trX_mask)), trX_bbx, trX_mask):
    #     img[int(round(bbx[3])):int(round(bbx[3])) + int(round(bbx[0])),
    #         int(round(bbx[2])):int(round(bbx[2])) + int(round(bbx[1])), :] = 1
    #     trX_mask[idx] = img

    # for idx, bbx, img in zip(xrange(len(valX_mask)), valX_bbx, valX_mask):
    #     img[int(round(bbx[3])):int(round(bbx[3])) + int(round(bbx[0])),
    #         int(round(bbx[2])):int(round(bbx[2])) + int(round(bbx[1])), :] = 1
    #     valX_mask[idx] = img

    # model = unet((512, 512, 3))
    # model.fit(trX, [trX_mask, trY], batch_size=128, nb_epoch=40,
    #           shuffle=True, validation_data=(valX, [valX_mask, valY]),
    #           verbose=1)
    # classes = {0: 'ALB', 1: 'BET', 2: 'DOL', 3: 'LAG', 4: 'NoF',
    #            5: 'OTHER', 6: 'SHARK', 7: 'YFT'}

    # df = pd.DataFrame(data=trX_bbox, columns=['image', 'class',  'x', 'y',
    #                                           'width', 'height'])
    # trX_bbx = df.iloc[:, 2:].values
    # del df, trX_bbox

    # trX = remove_mean(trX)
    # valX = remove_mean(valX)
    # teX = remove_mean(teX)

    # zca whitening
    # names = ['valX', 'trX', 'teX']
    # for dataset in range(3):
    #     for channel in range(3):
    #         tmp = eval(names[dataset])[:, :, :, channel].reshape(
    #             eval(names[dataset]).shape[0],
    #             np.prod(eval(names[dataset]).shape[1:3])
    #         )

    #         ZCA_valX = zca_whitening_matrix(tmp)

    #         eval(names[dataset])[:, :, :, channel] = np.dot(
    #             ZCA_valX,
    #             tmp
    #         ).reshape(tmp.shape[0], 299, -1)

    # model = SqueezeNet(weights=None, input_shape=(224, 224, 3))
    model = xception_model(shape=(299, 299), weights='imagenet')
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='bbox_loss', factor=0.2,
                                  patience=3, min_lr=1e-10,
                                  mode='auto', verbose=1)

    early_stopping = EarlyStopping(monitor='bbox_loss', patience=25,
                                   verbose=1)

    model.fit(trX, [trX_bbx, trY], batch_size=128, nb_epoch=40,
              shuffle=True, validation_data=(valX, [valX_bbx, valY]),
              verbose=1, callbacks=[reduce_lr, early_stopping])

    # scores = model.evaluate(valX, valY, verbose=0)
    # print('Validation loss:', scores[0])
    # print('Validation accuracy:', scores[1])

    y_val_pred = model.predict(valX)
    if len(y_val_pred) > 1:
        logLoss = log_loss(valY, y_val_pred[1])
    else:
        logLoss = log_loss(valY, y_val_pred)
    print("Log loss on valid set: => ", logLoss)

    y_pred = model.predict(teX)
    if len(y_pred) > 1:
        y_pred = np.clip(y_pred[1], 0.25, 0.975)
    else:
        y_pred = np.clip(y_pred, 0.25, 0.975)

    # test_scores = model.evaluate(teX, y_pred, verbose=0)
    # print('Test loss:', test_scores[0])
    # print('Test accuracy:', test_scores[1])

    # if logLoss < 1:
    df = pd.DataFrame(data=y_pred,
                      columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF',
                               'OTHER', 'SHARK', 'YFT'])

    df.insert(0, 'image', teX_id)
    df.to_csv('submit_bbox.csv', index=False)

    # trials = Trials()
    # objective(trials, trX, trY, valX, valY, teX)
