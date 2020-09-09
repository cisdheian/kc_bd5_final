from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def regresionLinear(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_predictions = lin_reg.predict(X)
    lin_mse = mean_squared_error(y, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y, lin_predictions)
    return lin_reg, lin_rmse, lin_mae


def arbolDecision(X, y):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X, y)
    tree_predictions = tree_reg.predict(X)
    tree_mse = mean_squared_error(y, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(y, tree_predictions)
    return tree_reg, tree_rmse, tree_mae


def bosqueAleatorio(X, y):
    forest_reg = RandomForestRegressor(n_estimators=100)
    forest_reg.fit(X, y)
    forest_predictions = forest_reg.predict(X)
    forest_mse = mean_squared_error(y, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = mean_absolute_error(y, forest_predictions)
    return forest_reg, forest_rmse, forest_mae


def kernelSVM(X, y):
    svm_reg = SVR(kernel="linear")
    svm_reg.fit(X, y)
    svr_predictions = svm_reg.predict(X)
    svr_mse = mean_squared_error(y, svr_predictions)
    svr_rmse = np.sqrt(svr_mse)
    svr_mae = mean_absolute_error(y, svr_predictions)
    return svm_reg, svr_rmse, svr_mae
