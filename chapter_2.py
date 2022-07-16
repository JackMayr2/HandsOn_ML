import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from scipy import stats


def chap_2_func():
    housing = pd.read_csv("datasets/housing/housing.csv")
    print(housing.head())
    print(housing["ocean_proximity"].value_counts())
    print(housing.describe())

    # Initial histograms of each parameter
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Creating categories in income so that the distribution in total data translates to our train and test sets as well
    # Displaying the distribution as a histogram as well
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    housing["income_cat"].hist()
    plt.show()

    # When creating train and test sets, maintain same distribution as raw data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Comparing distributions of test set and raw data
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    print(housing["income_cat"].value_counts() / len(housing))

    # Removing our income_cat column so data is back to normal
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # geographical representation of the data, able to see the outline of california
    housing.plot(kind="scatter", x="longitude", y="latitude")
    plt.show()

    # geographical representation with density as well, able to visually see cities of california
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

    # adding color to density map so that we can see the price of homes in addition to population density
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population",
                 figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()

    # Looking for correlations between variables and "median housing value
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Pandas Scatter matrix based off correlation found previously
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

    # Median income has the strongest correlation, so lets focus on that
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()

    # Creating new columns based on combinations of current columns
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    # Looking at our new correlations
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # creating our training data and labels
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Cleaning the data by dropping missing values
    # train_data.dropna()

    # replacing missing data with sklearn instead
    imputer = SimpleImputer(strategy="median")

    # Median can only be caluclated for numerical, so we should drop ocean_proximity
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    # We can work with ocean proximity as well, because it is categorical.
    # In this case, the data is also ordinal because generally
    # the closer a house is to the beach, the more expensive it is.
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(10))

    # Using the ordinal encoder from sklearn to change housing data from ordinal strings to numbers
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded[:10])

    # Ordinal is fine when there is a very clear progession like very bad, bad, good, great.
    # In this case, there is not this relationship, so a OneHot encoder can help to
    # still create numerical data, but without the necessary ordering.
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot

    # Creating a custom transformer allows you to modify hyperparameters such as
    # which columns are being analyzed and other things.
    # This is important, and I should do more research.
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    # Feature scaling is critical to machine learning, want attributes to be scaled similarly,
    # so one is not more or less wighted
    # creating a transformation pipeline helps to automate the process, o you don't have to custom build
    # scaling into each unique case
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)])

    housing_prepared = full_pipeline.fit_transform(housing)

    # Try out various models

    # First model: linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Measuring the MSE of a linear regression model
    housing_predictions = lin_reg.predict(housing_prepared)

    print(housing_labels)
    print(housing_predictions)

    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    # We can see that we are underfitting.
    # Lets try a different model

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # Time to evaluate our decision tree

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)

    # We get an error of 0.0, probably overfitting

    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)

    def display_scores(scores):
        print("Scores: ", scores)
        print("Mean: ", scores.mean())
        print("Standard Deviation: ", scores.std())

    display_scores(tree_rmse_scores)

    # Lets compare vs linear regression
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    # Lets try Random Forest

    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)

    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print(forest_rmse)

    scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    display_scores(forest_rmse_scores)

    # sklearn's GirdSearchCV optimizes hyperparameters for you

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    print(grid_search.best_params_)
    # Get the best estimator directly
    print(grid_search.best_estimator_)

    # Evaluation scores for grid search

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # when there are many hyperparameters to look at, used RandomizedSearchCV instead

    # Ensemble methods combine various models, and work especially well when the models make different types of errors
    feature_importances = grid_search.best_estimator_.feature_importances_
    print(feature_importances)

    # display according to attribute
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedtooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))

    # Running out model on the test set
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    # Confidence interval for the generalization error
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors))))
