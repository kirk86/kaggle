# coding: utf-8
"""
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''.

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.
"""

from __future__ import division
import numpy as np
# import load_data
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from pysofia.compat import RankSVMCV
from fastFM import sgd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# from sklearn.cross_validation import cross_val_score
from keras.utils import np_utils
import pandas as pd
from sklearn.svm import LinearSVC


def logloss(y_true, y_pred, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(y_pred, epsilon, 1.0-epsilon)
    return - np.mean(y_true * np.log(y_pred) +
                     (1.0 - y_true) * np.log(1.0 - attempt))


def ensemble(trX, trY, teX, teX_id):

    np.random.seed(2017)  # seed to shuffle the train set

    n_folds = 10
    # verbose = True
    shuffle = False

    # X, y, X_test = load_data.load()
    # X_submission is X_test
    trX = trX.reshape(trX.shape[0], np.prod(trX.shape[1:]))
    trY = np.argmax(trY, axis=1)
    teX = teX.reshape(teX.shape[0], np.prod(teX.shape[1:]))

    if shuffle:
        idx = np.random.permutation(trY.size)
        trX = trX[idx]
        trY = trY[idx]

    skf = StratifiedKFold(n_splits=n_folds)

    clfs = [
        GaussianProcessClassifier(kernel=RationalQuadratic,
                                  n_restarts_optimizer=100,
                                  max_iter_predict=1000,
                                  warm_start=True,
                                  n_jobs=-1),
        RandomForestClassifier(n_estimators=1000, n_jobs=-1,
                               criterion='gini'),
        RandomForestClassifier(n_estimators=1000, n_jobs=-1,
                               criterion='entropy'),
        ExtraTreesClassifier(n_estimators=1000, n_jobs=-1,
                             criterion='gini'),
        ExtraTreesClassifier(n_estimators=1000, n_jobs=-1,
                             criterion='entropy'),
        XGBClassifier(learning_rate=0.05, subsample=0.5,
                      max_depth=6, n_estimators=1000),
        LGBMClassifier(learning_rate=0.05, subsample=0.5,
                       max_depth=6, n_estimators=1000),
        KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree',
                             p=1, leaf_size=30, n_jobs=-1),
        QuadraticDiscriminantAnalysis(reg_param=1e-2),
        LinearSVC(class_weight='auto', verbose=True, max_iter=10000,
                  tol=1e-6, C=1)
        # RankSVMCV(max_iter=500)
    ]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((trX.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((teX.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((teX.shape[0], n_folds))
        for i, (trX_idx, valX_idx) in zip(range(n_folds),
                                          skf.split(np.zeros(len(trX)),
                                                    trY)):
            print "Fold", i
            X = trX[trX_idx]
            Y = trY[trX_idx]
            valX = trX[valX_idx]
            valY = trY[valX_idx]
            clf.fit(X, Y)
            valY_pred = clf.predict_proba(valX)
            dataset_blend_train[valX_idx, j] = valY_pred[:, 1]
            dataset_blend_test_j[:, i] = clf.predict_proba(teX)[:, 1]
            # print(metrics.classification_report(valY,
            #                                     np.argmax(valY_pred, axis=1)))
            # print(metrics.confusion_matrix(valY, np.argmax(valY_pred, axis=1)))
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)  # averaging
                                                                 # the
                                                                 # predictions

    print
    print "Blending."
    clf = LogisticRegression(C=10, solver='lbfgs', penalty='l2',
                             multi_class='multinomial', n_jobs=-1)
                                # Creating model S like the kaggle
                                # blog example on stacking. Is
                                # ensembling same as stacking? Take
                                # away message, in stacking we use the
                                # predictions of the base models as
                                # features(i.e. meta features) for the
                                # stakced model. The stacked model
                                # able to dicern where each model
                                # performs well and where poorly.
                                # It’s also important to note that the
                                # meta features in row i of train_meta
                                # are not dependent on the target
                                # value in row i because they were
                                # produced using information that
                                # excluded the target_i in the base
                                # models’ fitting procedure.

    clf.fit(dataset_blend_train, trY)
    # y_pred = clf.predict_proba(dataset_blend_test)[:, 1]
    y_pred = clf.predict_proba(dataset_blend_test)

    print "Linear stretch of predictions to [0,1]"
    y_pred = (y_pred - y_pred.min()) \
        / (y_pred.max() - y_pred.min())

    # print("Log loss emanuele = {}, sklearn = {}"
    #       .format(logloss(trY, y_pred), metrics.log_loss(trY, y_pred)))

    print "Saving Results."
    df = pd.DataFrame(y_pred, columns=['ALB', 'BET', 'DOL', 'LAG',
                                       'NoF', 'OTHER', 'SHARK', 'YFT'])
    df.insert(0, 'image', teX_id)
    df.to_csv('submit_ensemble.csv', index=False)
    # tmp = np.vstack([range(1, len(y_pred)+1), y_pred]).T
    # np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
    #            header='image,ALB,BET,LOG,NoF,YFT,SHARK,OTHER', comments='')
