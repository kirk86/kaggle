# import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
# from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback


def build_top_layer(model, deeper=False, start=11):
    if deeper:
        top = Sequential([
            # top classification layers
            BatchNormalization(input_shape=model.output_shape[1:], axis=1),
            # Block 4
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            BatchNormalization(axis=1),
            # output = MaxPooling2D((2, 2), strides=(2, 2), name='bl4_pool')(output)
            Dropout(0.5),
            # b
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2), strides=(2, 2)),
            Dropout(0.3),
            Flatten(name='flatten'),
            Dense(4096, activation='relu'),
            BatchNormalization(axis=1),
            Dropout(0.2),
            Dense(4096, activation='relu'),
            BatchNormalization(axis=1),
            Dropout(0.2),
            Dense(8, activation='softmax', name='predictions')
            ])
    else:
        top = Sequential([
            BatchNormalization(input_shape=model.output_shape[1:],
                               axis=1),
            Flatten(),
            Dense(4096, activation='relu'),
            BatchNormalization(axis=1),
            Dropout(0.3),
            Dense(4096, activation='relu'),
            BatchNormalization(axis=1),
            Dense(8, activation='softmax')
           ])

    return top


def finetune(model, shape, deeper=False):
    # ---------- bottleneck features example starts here ----------
    # datagen = ImageDataGenerator(rescale=1./255)
    # generator = datagen.flow_from_directory(
    #     'data/cropped_train',
    #     target_size=shape,
    #     batch_size=64,
    #     class_mode=None,  # this means our generator will only yield
    #                       # batches of data, no labels
    #     shuffle=False)  # our data will be in order, so all first 1000
    #                     # images will be cats, then 1000 dogs the
    #                     # predict_generator method returns the output
    #                     # of a model, given a generator that yields
    #                     # batches of numpy data
    # bottleneck_features_train = model.predict_generator(generator, 2904)
    # # save the output as a Numpy array
    # np.save(open('bottleneck_features_train.npy', 'w'),
    #         bottleneck_features_train)

    # generator = datagen.flow_from_directory('data/validation',
    #                                         target_size=shape,
    #                                         batch_size=64,
    #                                         class_mode=None,
    #                                         shuffle=False)

    # bottleneck_features_validation = model.predict_generator(generator, 500)

    # np.save(open('bottleneck_features_validation.npy', 'w'),
    #         bottleneck_features_validation)

    # train_data = np.load(open('bottleneck_features_train.npy'))
    # # the features were saved in order, so recreating the labels is
    # # easy
    # train_labels = to_categorical(np.array([0] * 1266 + [1] * 169 +
    #                                        [2] * 87 + [3] * 53 +
    #                                        [4] * 344 + [5] * 243 +
    #                                        [6] * 141 + [7] * 601))

    # validation_data = np.load(open('bottleneck_features_validation.npy'))
    # validation_labels = to_categorical(np.array([0] * 217 + [1] * 21 +
    #                                             [2] * 18 + [3] * 13 +
    #                                             [4] * 54 + [5] * 44 +
    #                                             [6] * 29 + [7] * 104))

    # # set the first 25 layers (up to the last conv block)
    # # to non-trainable (weights will not be updated)
    # for layer in model.layers:
    #     layer.trainable = False

    # top1 = build_top_layer(model, deeper)

    # top1.compile(optimizer='rmsprop',
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])

    # top1.fit(train_data, train_labels,
    #          nb_epoch=10, batch_size=300,
    #          validation_data=(validation_data, validation_labels))

    # top1.save_weights('bottleneck_fc_model.h5')
    # del top1
    # ------- bottleneck features example ends here ---------------

    # ----------- finetuning example starts here -----------------
    # build a classifier model to put on top of the convolutional model
    top2 = build_top_layer(model, deeper)

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top2.load_weights('bottleneck_fc_model.h5')

    # add the model on top of the convolutional base
    # final_model = Model(model.input, top2.output)
    final_model = Sequential([
        model,
        top2
    ])

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=SGD(lr=1e-4, momentum=0.9),
                        metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
           rescale=1./255,
           shear_range=0.2,
           zoom_range=0.2,
           horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
           'data/cropped_train',
           target_size=shape,
           batch_size=300,
           class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
           'data/validation',
           target_size=shape,
           batch_size=300,
           class_mode='categorical')

    # best_model = ModelCheckpoint(final_model.name + '_weights.h5',
    #                              monitor='val_loss', verbose=1,
    #                              save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=1e-10,
                                  mode='auto', verbose=1)

    save_model = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                final_model.save(final_model.name +
                                                 '_model.h5'))
    model_weights = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                   final_model.save_weights('model_weights.h5'))

    # fine-tune the model
    final_model.fit_generator(train_generator,
                              samples_per_epoch=8000,
                              nb_epoch=10,
                              validation_data=validation_generator,
                              nb_val_samples=800,
                              callbacks=[reduce_lr, save_model, model_weights])
    # ----------- finetuning example ends here ------------------

    return final_model
