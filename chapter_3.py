import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score, precision_recall_curve, roc_curve, roc_auc_score

import requests
import ssl


def chap_3_func():
    # Bypass SSL certificates, don't completely understand but it works
    requests.packages.urllib3.disable_warnings()
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    mnist = fetch_openml('mnist_784', version=1)
    mnist.keys()
    x, y = np.array(mnist["data"]), np.array(mnist['target'])
    y = [int(numeric_string) for numeric_string in y]
    y = np.array(y)
    print('X Shape: ', x.shape)
    print('----------------------------------------------------', '\n')
    print('Y Shape:', y.shape)
    print('----------------------------------------------------', '\n')

    # Look at the data we are working with

    some_digit = x[0]
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")
    plt.show()

    # Splitting data into train and test
    X_train, X_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    print(y_train.shape)
    print(y_test.shape)

    # Creating a binary classifier for whether the image is 5 or not 5
    y_train_5 = np.array((y_train == 5))
    y_test_5 = np.array((y_test == 5))

    print(y_train_5.shape)
    print(y_test_5.shape)

    print(np.unique(y_train_5))

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    print(sgd_clf.predict([some_digit]))

    # Classifiers are trickier to evaluate than regressors
    # First we will use cross-validation
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

    print('Cross validation score: ', cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    print('----------------------------------------------------', '\n')

    # Performs at 95%!  but context is everything. 90% of images are not 5,
    # therefore we would expect a bad estimator to still perform at 90%.
    # Lets looks at a confusion matrix to get a better idea of how our classification method does
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print('confusion matrix:', confusion_matrix(y_train_5, y_train_pred))
    print('----------------------------------------------------', '\n')

    # We can also get the precision and recall from our classifier
    print('precision score: ', precision_score(y_train_5, y_train_pred))
    print('----------------------------------------------------', '\n')

    print('recall score:', recall_score(y_train_5, y_train_pred))
    print('----------------------------------------------------', '\n')

    # F1 score is a way of combining the two metrics
    print('f1 score: ', f1_score(y_train_5, y_train_pred))
    print('----------------------------------------------------', '\n')

    # We can change the threshold, in turn affecting our precision/recall tradeoff
    # The threshold is like the confidence of the classifier, so if it isn't as confident
    # as the threshold value, it will return false
    y_scores = sgd_clf.decision_function([some_digit])
    print('y scores: ', y_scores)
    print('----------------------------------------------------', '\n')

    threshold = 0

    y_some_digit_pred = (y_scores > threshold)
    print(y_some_digit_pred)

    threshold = 8000
    y_some_digit_pred = (y_scores > threshold)
    print(y_some_digit_pred)

    # If we look at the decision scores, we can decide what the best threshold for our
    # application would be
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

    print(y_scores.shape)
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    def plot_precision_recall_vs_threshold(prec, rec, thresh):
        plt.plot(thresh, prec[:-1], "b--", label="Precision")
        plt.plot(thresh, rec[:-1], "g--", label="Recall")
        plt.xlabel('Threshold')
        plt.legend(loc='best')

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

    def plot_precision_against_recall(prec, rec):
        plt.plot(rec[:-1], prec[:-1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')

    plot_precision_against_recall(precisions, recalls)
    plt.show()

    # If we only care about having 90% precision
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

    y_train_pred_90 = (y_scores >= threshold_90_precision)

    print('Precision Score: ', precision_score(y_train_5, y_train_pred_90))
    print('----------------------------------------------------', '\n')

    print('Recall Score: ', recall_score(y_train_5, y_train_pred_90))
    print('----------------------------------------------------', '\n')

    # ROC curve common for binary classifiers, plots true positive rate against false positive rate

    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')

    plot_roc_curve(fpr, tpr)
    plt.show()

    print('ROC AUC Score: ', roc_auc_score(y_train_5, y_scores))
    print('----------------------------------------------------', '\n')

    # Building a random forest classifier to compare to the SGD classifier
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

    y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    # plot ROC curves against each other
    plt.plot(fpr, tpr, "b:", label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, label="Random Forest")
    plt.legend(loc="best")
    plt.show()

    # Lets try to classify these numbers in a multiclass classification way instead of binary

    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    svm_clf.predict([some_digit])

    sgd_clf.fit(X_train, y_train)
    print('multi-class sgd classifier cross val score: ',
          cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))
    print('----------------------------------------------------', '\n')

    # We can scale the inputs to improve this
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    print('multi-class sgd classifier wtih standard scaler cross val score: ',
          cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))
    print('----------------------------------------------------', '\n')

    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    # to see where our function makes errors, we can color on the diagonal
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums

    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

    # multilabel classification is used when there are multiple features in the same image
    # that you want to account for (think multiple faces)
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)

    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    print('average multilabel f1 score', f1_score(y_multilabel, y_train_knn_pred, average='macro'))