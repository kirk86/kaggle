from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D, ZeroPadding2D
from keras.layers import Input, GlobalAveragePooling2D, Merge
from keras.layers import TimeDistributed, LSTM, SimpleRNN, BatchNormalization
from keras.initializations import normal, identity
from keras.optimizers import SGD, RMSprop
from keras.constraints import maxnorm


nb_classes = 8


def cnn_model(shape, nb_neurons=8, lr_rate=1e-2, momentum=0.9,
              init_weights='uniform', activation='relu',
              weight_constraint=0, dropout_rate=0):

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=shape + (3,)))

    model.add(Convolution2D(4, 3, 3, init=init_weights, activation=activation,
                            W_constraint=maxnorm(weight_constraint)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(4, 3, 3, init=init_weights, activation=activation,
                            W_constraint=maxnorm(weight_constraint)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(8, 3, 3, init=init_weights, activation=activation,
                            W_constraint=maxnorm(weight_constraint)))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(8, 3, 3, init=init_weights, activation=activation,
                            W_constraint=maxnorm(weight_constraint)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(nb_neurons, init=init_weights, activation=activation,
                    W_constraint=maxnorm(weight_constraint)))

    model.add(Dropout(dropout_rate))

    model.add(Dense(nb_neurons, init=init_weights, activation=activation,
                    W_constraint=maxnorm(weight_constraint)))

    model.add(Dropout(dropout_rate))
    model.add(Dense(nb_classes, init=init_weights, activation='softmax'))

    sgd = SGD(lr=lr_rate, decay=1e-6, momentum=momentum, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def vgg16_model(shape, weights=None):
    from keras.applications import VGG16
    model = VGG16(include_top=False, weights=weights,
                  input_tensor=Input(shape=shape + (3,)))
    output = model.output
    output = Flatten(name='flatten')(output)
    output = Dense(4096, activation='relu', name='fc1')(output)
    output = Dropout(0.35)(output)
    output = Dense(4096, activation='relu', name='fc2')(output)
    output = Dropout(0.5)(output)
    output = Dense(nb_classes, activation='softmax',
                   name='predictions')(output)
    final_model = Model(model.input, output)

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    final_model.compile(optimizer=sgd, loss='categorical_crossentropy',
                        metrics=['accuracy'])

    print("Model => vgg16")

    return final_model


def vgg19_model(shape, weights=None, deeper=False, start=11):
    from keras.applications import VGG19
    model = VGG19(include_top=False, weights=weights,
                  input_tensor=Input(shape=shape + (3,)))

    if deeper:
        # removing from block3_conv4 ... onwards
        for idx in range(start, len(model.layers)):
            model.layers.pop()

    # output = model.output
    # output = Flatten(name='flatten')(output)
    # output = Dense(4096, activation='relu', name='fc1')(output)
    # output = Dropout(0.5)(output)
    # output = Dense(4096, activation='relu', name='fc2')(output)
    # output = Dropout(0.5)(output)
    # output = Dense(nb_classes, activation='softmax',
    # name='predictions')(output)
    # final_model = Model(model.input, output)

    # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    # final_model.compile(optimizer=sgd,
    #                     loss='categorical_crossentropy',
    #                     metrics=['accuracy'])

    print("Model => vgg19")

    return model


def inceptionv3_model(shape, weights=None):
    from keras.applications import InceptionV3
    model = InceptionV3(include_top=False, weights=weights,
                        input_tensor=Input(shape=shape + (3,)))
    output = model.output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(nb_classes, activation='softmax', name='predictions')(output)
    final_model = Model(model.input, output)

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    final_model.compile(optimizer=sgd, loss='categorical_crossentropy',
                        metrics=['accuracy'])

    print("Model => inception-v3")

    return final_model


def resnet50_model(shape, weights=None):
    from keras.applications import ResNet50
    model = ResNet50(include_top=False, weights=weights,
                     input_tensor=Input(shape=shape + (3,)))

    x = model.output
    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    final_model = Model(model.input, x)

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    final_model.compile(optimizer=sgd,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    print("Model => resnet")

    return final_model


def xception_model(shape, weights=None):
    from keras.applications import Xception
    model = Xception(include_top=False, weights=weights,
                     input_tensor=Input(shape=shape + (3,)))

    output = model.output
    output = BatchNormalization(axis=3)(output)
    output = GlobalAveragePooling2D(name='global_avg_pool')(output)
    output = BatchNormalization(axis=1)(output)
    output = Dropout(0.5)(output)
    output = BatchNormalization(axis=1)(output)
    output_bbox = Dense(4, name='bbox')(output)
    output_class = Dense(nb_classes, activation='softmax',
                         name='predictions')(output)
    # output = Dense(nb_classes, activation='softmax',
    #                name='predictions')(output)
    final_model = Model(model.input, [output_bbox, output_class])
    # final_model = Model(model.input, output)

    # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    final_model.compile(optimizer='adamax',
                        loss=['msle', 'categorical_crossentropy'],
                        metrics=['accuracy', 'bbox_accuracy'],
                        loss_weights=[0.001, 1.])
    # final_model.compile(optimizer=sgd,
    #                     loss=['categorical_crossentropy'],
    #                     metrics=['accuracy'])
    print("Model => xception")

    return final_model


def vgg19_plus_vgg16_model(shape, weights=None):
    from keras.applications import VGG19
    model_vgg19 = VGG19(include_top=False, weights=weights,
                        input_tensor=Input(shape=shape + (3,)))

    from keras.applications import VGG16
    model_vgg16 = VGG16(include_top=False, weights=weights,
                        input_tensor=Input(shape=shape + (3,)))

    left = Sequential()
    right = Sequential()
    final = Sequential()

    left.add(model_vgg19)
    left.add(Flatten(name='vgg19_flatten'))
    right.add(model_vgg16)
    right.add(Flatten(name='vgg16_flatten'))
    final.add(Merge([left, right], mode='concat'))
    final.add(Dense(4096, activation='relu', name='fc1'))
    final.add(Dropout(0.35, name='dropout1'))
    final.add(Dense(4096, activation='relu', name='fc2'))
    final.add(Dropout(0.5, name='dropout2'))
    final.add(Dense(nb_classes, activation='softmax', name='predictions'))
    final.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    print("Model => vgg19 + vgg16")

    return final


def hierarchical_rnn(shape):
    # embedding dimensions
    row_hidden = 128
    col_hidden = 128

    x = Input(shape=shape+(3,))
    # encode a row of pixels using TimeDistributed wrapper
    encoded_rows = TimeDistributed(LSTM(output_dim=row_hidden))(x)
    # encode columns of encoded_rows
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    # final prediction and model
    prediction = Dense(nb_classes, activation='softmax')(encoded_columns)
    model = Model(input=x, output=prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def irnn(shape_orig):
    hidden_units = 100
    learning_rate = 1e-6

    model = Sequential()
    model.add(SimpleRNN(output_dim=hidden_units,
                        init=lambda shape, name:
                        normal(shape, scale=0.001, name=name),
                        inner_init=lambda shape, name:
                        identity(shape, scale=1.0, name=name),
                        activation='relu',
                        input_shape=shape_orig))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    return model


def shallow_model(shallow_type):
    if shallow_type == 'regression':
        from sklearn.linear_model import LogisticRegression
        # fit a logistic regression model
        return LogisticRegression(verbose=1)

    if shallow_type == 'mnb':
        from sklearn.naive_bayes import MultinomialNB
        # fit a Naive Bayes model to the data
        # return GaussianNB(class_count=array shape=> (n_classes,))
        return MultinomialNB()

    if shallow_type == 'tree':
        from sklearn.tree import DecisionTreeClassifier
        # fit a CART model to the data
        return DecisionTreeClassifier(verbose=1)

    if shallow_model == 'svc':
        from sklearn.svm import SVC
        # from sklearn.svm import libsvm
        # from sklearn.svm import NuSVC
        # fit a SVM model to the data
        return SVC(C=0.001, gamma='auto', class_weigth='balanced',
                   probability=True, decision_function_shape='ovr',
                   cache_size=200)

    if shallow_type == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=800, max_depth=35,
                                      min_samples_split=15,
                                      min_samples_leaf=10,
                                      max_features='sqrt', n_jobs=-1,
                                      verbose=1)

    if shallow_type == 'extratree':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_jobs=-1, verbose=1)

    if shallow_type == 'ensemble':
        from sklearn.ensemble import AdaBoostClassifier
        # from sklearn.ensemble import BaggingClassifier
        # from sklearn.ensemble import GradientBoostingClassifier
        # from sklearn.ensemble import RandomTreesEmbedding
        # from slkearn.ensemble import VotingClassifier
        # VotingClassifier(clf1, clf2, clf3)
        # BaggingClassifier(n_jobs=-1)
        # GradientBoostingClassifier()
        # IsolationForest()
        # RandomTreesEmbedding(n_jobs=-1)
        return AdaBoostClassifier(verbose=1)

    if shallow_type == 'gaussian':
        from sklearn.gaussian_process import GaussianProcessClassifier
        return GaussianProcessClassifier(n_jobs=-1)

    if shallow_type == 'linear':
        # from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import RandomizedLogisticRegression
        return RandomizedLogisticRegression(n_jobs=-1, verbose=1)
        # SGDClassifier(n_jobs=-1)

    if shallow_type == 'multiclass':
        from sklearn.multiclass import OneVsRestClassifier
        # from sklearn.multioutput import MultiOutputClassifier
        # MultiOutputClassifier(n_jobs=-1)
        return OneVsRestClassifier(n_jobs=-1, verbose=1)

    if shallow_type == 'kNN':
        from sklearn.neighbors import KNeighborsClassifier
        # from sklearn.neighbors import RadiusNeighborsClassifier
        # RadiusNeighborsClassifier()
        return KNeighborsClassifier(n_jobs=-1, verbose=1)

    if shallow_type == 'xgboost':
        import xgboost
        return xgboost.XGBClassifier(nthread=8)
