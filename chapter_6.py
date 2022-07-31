import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt


def chap_6_func():
    iris = load_iris()
    X = iris.data[:, 2:]
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

    sklearn.tree.plot_tree(tree_clf,
                           feature_names=iris.feature_names[2:],
                           class_names=iris.target_names,
                           filled=True)
    plt.show()

    tree_reg = DecisionTreeRegressor(max_depth=2)
    tree_reg.fit(X, y)

    sklearn.tree.plot_tree(tree_reg,
                           feature_names=iris.feature_names[2:],
                           class_names=iris.target_names,
                           filled=True)
    plt.show()


