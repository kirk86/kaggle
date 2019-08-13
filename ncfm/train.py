import numpy as np
# import multiprocessing
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard, ReduceLROnPlateau
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn.metrics import log_loss
from preprocess import process_train, process_validation
from finetune import finetune


np.random.seed(2017)
nb_classes = 8


def augment_train(shape):
    # dimensions of our images.
    img_width, img_height = shape

    labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    train_dir = 'data/cropped_train'
    valid_dir = 'data/validation'

    nb_train_samples = 5000
    nb_valid_samples = 1000

    batch_size = 200

    # nb_epoch = 30

    # augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       rotation_range=10.,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True,
                                       vertical_flip=True)
                                       #fill_mode='nearest'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        #save_to_dir='temp/visualization/train',
        #save_prefix='aug_img_',
        classes=labels,
        class_mode='categorical'
    )

    # augmentation configuration for validation, only rescaling
    valid_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        #save_to_dir='temp/visualization/validation',
        #save_prefix='aug_img_',
        classes=labels,
        class_mode='categorical'
    )

    return (train_generator, nb_train_samples), \
        (validation_generator, nb_valid_samples)


def construct_model(shape=(299, 299), model_type='cnn', shallow_type='svc'):

    if model_type == 'shallow':
        from models import shallow_model
        model = shallow_model(shallow_type)

    if model_type == 'cnn':
        from models import cnn_model
        model = cnn_model(shape)

    if model_type == 'vgg16':
        from models import vgg16_model
        model = vgg16_model(shape, weights='imagenet')

    if model_type == 'vgg19':
        from models import vgg19_model
        model = vgg19_model(shape, weights='imagenet', deeper=False)

    if model_type == 'inceptionv3':
        from models import inceptionv3_model
        model = inceptionv3_model(shape, weights='imagenet')

    if model_type == 'resnet':
        from models import resnet50_model
        model = resnet50_model(shape, weights='imagenet')

    if model_type == 'xception':
        from models import xception_model
        model = xception_model(shape, weights='imagenet')

    if model_type == 'hrnn':
        from models import hierarchical_rnn
        model = hierarchical_rnn(shape)

    if model_type == 'irnn':
        from models import irnn
        model = irnn(shape)

    if model_type is not 'shallow':
        print(model.summary())

    return model, shape, model_type


def get_val_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def train_model_cv(shape, model_type, shallow_type, nfolds=10):
    # input image dimensions
    # batch_size = 64
    nb_epoch = 1
    random_state = 51

    trX, trY, trX_id = process_train(shape)

    yfull_train = dict()

    # teX, teX_id = process_test(shape)

    # from ensemble import ensemble
    # ensemble(trX, trY, teX)

    # return

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True,
                          random_state=random_state)

    num_fold = 0
    sum_score = 0
    models = []

    for trX_idx, valX_idx in skf.split(np.zeros(len(trX)), np.nonzero(trY)[1]):
        # construct correct data format regarding the model used
        if model_type == 'irnn':
            X = trX[trX_idx].reshape(trX[trX_idx].shape[0], -1, 1)
            Y = trY[trX_idx]
            valX = trX[valX_idx].reshape(trX[valX_idx].shape[0], -1, 1)
            valY = trY[valX_idx]
            shape = X.shape[1:]
        if model_type == 'shallow':
            X = trX[trX_idx].reshape(trX[trX_idx].shape[0],
                                     np.prod(trX[trX_idx].shape[1:]))
            Y = np.nonzero(trY[trX_idx])[1]
            valX = trX[valX_idx].reshape(trX[valX_idx].shape[0],
                                         np.prod(trX[valX_idx].shape[1:]))
            valY = np.nonzero(trY[valX_idx])[1]
        else:  # deep NN models
            X = trX[trX_idx]
            Y = trY[trX_idx]
            valX = trX[valX_idx]
            valY = trY[valX_idx]

        model, shape, model_spec = construct_model(shape,
                                                   model_type,
                                                   shallow_type)

        (train_generator, nb_train_samples), \
            (validation_generator, nb_valid_samples) = augment_train(shape)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X), len(Y))
        print('Split valid: ', len(valX), len(valY))

        if model_spec == 'shallow':   # sklearn models
            model.fit(X, Y)
            valY_pred = model.predict_proba(valX)
            print(metrics.classification_report(valY,
                                                np.argmax(valY_pred, axis=1)))
            print(metrics.confusion_matrix(valY, np.argmax(valY_pred, axis=1)))

        else:                       # NN models
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=5, verbose=1)

            best_model = ModelCheckpoint('./' + model_type + '_weights.h5',
                                         monitor='val_loss', verbose=1,
                                         save_best_only=True)

            tensorboard = TensorBoard(log_dir='./logs', histogram_freq=4,
                                       write_graph=True, write_images=False)

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=3, min_lr=1e-10,
                                          mode='auto', verbose=1)

            # model.fit(X, Y, batch_size=batch_size,
            #           nb_epoch=nb_epoch, shuffle=True, verbose=1,
            #           validation_data=(valX, valY), callbacks=callbacks)

            model.fit_generator(train_generator,
                                samples_per_epoch=nb_train_samples,
                                nb_epoch=nb_epoch,
                                validation_data=validation_generator,
                                nb_val_samples=nb_valid_samples,
                                callbacks=[best_model, early_stopping,
                                           tensorboard, reduce_lr])

            # valY_pred = model.predict(valX.astype('float32'),
            #                           batch_size=batch_size,
            #                           verbose=1)

            valY_pred = model.predict_generator(validation_generator,
                                                nb_valid_samples)

        score = log_loss(valY, valY_pred)
        print('Score log_loss: ', score)
        sum_score += score * len(valX_idx)

        # Store valid predictions
        for i in range(len(valX_idx)):
            yfull_train[valX_idx[i]] = valY_pred[i]

        models.append(model)

    # from gridsearch import gridsearch
    # gridsearch(trX, trY, model)

    # from gridsearch import randomsearch
    # randomsearch(model, trX, trY)

    score = sum_score/len(trX)
    print("Log_loss train independent avg: ", score)

    info_string = str(model_spec) + '_loss_' + str(score) \
        + str(nfolds) + '_ep_' + str(nb_epoch)

    return info_string, models, shape


def train_model(shape, model_type, shallow_type, nfolds=10, tuning=False):
    # mean_pixel = [103.939, 116.779, 123.68]
    # img = img.astype(np.float32, copy=False)
    # for c in range(3):
    #     img[:, :, c] = img[:, :, c] - mean_pixel[c]

    # input image dimensions
    # batch_size = 64
    nb_epoch = 10

    model, shape, model_spec = construct_model(shape,
                                               model_type,
                                               shallow_type)
    if tuning:
        model = finetune(model, shape)
    else:
        B
        valX, valY, valX_idx = process_validation(shape)
        trX, trY, trX_idx = process_train(shape)

        # valY_pred = model.predict_proba(valX)
        # print(metrics.classification_report(valY,
        #                                     np.argmax(valY_pred, axis=1)))
        # print(metrics.confusion_matrix(valY, np.argmax(valY_pred, axis=1)))

        # NN models
        # early_stopping = EarlyStopping(monitor='val_loss',
        #                                patience=5, verbose=1)

        best_model = ModelCheckpoint(model_type + '_weights.h5',
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True)

        # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=4,
        #                           write_graph=True, write_images=False)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=1e-10,
                                      mode='auto', verbose=1)

        (train_generator, nb_train_samples), \
            (validation_generator, nb_valid_samples) = augment_train(shape)

        model.fit_generator(train_generator,
                            samples_per_epoch=nb_train_samples,
                            nb_epoch=nb_epoch,
                            validation_data=validation_generator,
                            nb_val_samples=nb_valid_samples,
                            callbacks=[reduce_lr, best_model])
                            # callbacks=[best_model, early_stopping,
                            #            tensorbard, reduce_lr])

    valY_pred = model.predict(valX)

    sum_score = 0
    score = log_loss(valY, valY_pred)
    print('Score log_loss: ', score)
    sum_score += score * len(valX)
    print('Sum Score log_loss: ', sum_score)

    score = sum_score/len(trX)
    print("Log_loss train independent avg: ", score)

    info_string = '_loss_' + str(score) \
        + str(nfolds) + '_ep_' + str(nb_epoch)

    return info_string, model, shape
