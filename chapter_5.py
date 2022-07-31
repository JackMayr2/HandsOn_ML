import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.datasets import make_moons
import plotly.graph_objects as go  # for data visualization
import plotly.express as px  # for data visualization


def chap_5_func():
    def plot_svms(title, type_clf):
        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            return xx, yy

        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, **params)
            return out

        fig, ax = plt.subplots()

        # Set-up grid for plotting.
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, type_clf, xx, yy, cmap=plt.cm.PRGn, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.PRGn, s=20, edgecolors='k')
        ax.set_ylabel('X2')
        ax.set_xlabel('X1')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.legend()
        plt.show()

    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = (iris["target"] == 2).astype(np.float64)

    svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
    svm_clf.fit(X, y)
    plot_svms("linear SVM", svm_clf)

    # we can add features to make something separable
    X, y = make_moons(n_samples=100, noise=0.15)
    polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])
    polynomial_svm_clf.fit(X, y)
    plot_svms("polynomial SVM", polynomial_svm_clf)

    # we can use polynomial kernel to train higher dimensional data.
    # the kernel trick ensures our algorithm isn't dramatically slowed
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X, y)
    plot_svms("polynomial kernel SVM", poly_kernel_svm_clf)

    # we can also perform regressions with svms
    svm_reg = LinearSVR(epsilon=1.5)
    svm_reg.fit(X, y)

    # if the data is not linear
    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)
