import numpy as np
import datetime
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from preprocess import process_test

np.random.seed(2017)
nb_classes = 8


def augment_test(shape):
    img_width, img_height = shape
    test_dir = 'test'

    nb_test_samples = 1000
    batch_size = 100

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,  # Important
        seed=np.random.randint(0, 100000 + 1),
        classes=None,
        class_mode=None
    )

    return (test_generator, nb_test_samples)


def submit(predictions, test_id, info):
    df = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL',
                                            'LAG', 'NoF', 'OTHER',
                                            'SHARK', 'YFT'])
    # df.loc[:, 'image'] = pd.Series(test_id, index=df.index)
    df.insert(0, 'image', test_id)
    now = datetime.datetime.now()
    file_csv = 'submission_' + info + '_' + \
               str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    df.to_csv(file_csv, index=False)


def dict2list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def test_model_cv(info_string, models, shape, model_type):
    # batch_size = 128
    num_fold = 0
    yfull_test = []
    teX_id = []
    nfolds = len(models)

    teX, teX_id = process_test(shape)

    for fold in range(nfolds):
        model = models[fold]
        num_fold += 1
        print('Start Test KFold number {} from {}'.format(num_fold, nfolds))

        (test_generator, nb_test_samples) = augment_test(shape)

        if model_type == 'irnn':
            teX = teX.reshape(teX.shape[0], -1, 1)
        if model_type == 'shallow':
            teX = teX.reshape(teX.shape[0], np.prod(teX.shape[1:]))
            teY_pred = model.predict_proba(teX)
        else:
            # teY_pred = model.predict(teX, batch_size=batch_size,
            #                          verbose=1)
            teY_pred = model.predict_generator(test_generator, nb_test_samples)

        yfull_test.append(teY_pred)

    result = merge_folds_mean(yfull_test, nfolds)

    info_string = info_string + '_folds_' + str(nfolds)
    if np.dim(result) < 2:
        result = np_utils.to_categorical(result, nb_classes)

    submit(result, teX_id, info_string)


def test_model(info_string, model, shape, model_type):
    # batch_size = 128
    teX, teX_id = process_test(shape)

    (test_generator, nb_test_samples) = augment_test(shape)

    # model = load_model(path_to_weights.h5)
    teY_pred = model.predict_generator(test_generator, nb_test_samples)

    result = teY_pred

    info_string = info_string

    if average:
        print("Loading model and weights...")
        for idx in range(nb_augmentation):
            print("{}th augmentation for testing...".format(idx))
            random_seed = np.random.randint(0, 100000 + 1)
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(img_widht, img_height),
                batch_size=batch_size,
                B
                shuffle=False,
                seed=random_seed,
                classes=None,
                class_mode=None)

            test_image_filenames = test_generator.filenames
            print("Beging to predict for testing data...")
            if idx == 0:
                teY_pred = model.predict_generator(test_generator, nb_test_samples)
            else:
                teY_pred = model.predict_generator(test_generator, nb_test_samples)

        teY_pred /= nb_augmentation


    # if np.dim(result) < 2:
    #     result = np_utils.to_categorical(result, nb_classes)

    submit(result, teX_id, info_string)
