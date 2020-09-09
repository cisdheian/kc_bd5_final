from Utilities import getDF
from DataSetTransformation import dataSetSplit, dataSetPreparation, dataSetPipelineTransform
from RegressionModel import regresionLinear, arbolDecision, bosqueAleatorio, kernelSVM
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Bajar el datasource
listing = getDF("listings.csv.gz", "listings.csv")
print(f'Dimensiones del dataset completo: {listing.shape}')
dataSetSplit(listing)

# Train
listing = pd.read_csv('train.csv', sep=';', decimal='.')
listing = dataSetPreparation(listing)
listing_prepared, listing_label = dataSetPipelineTransform(listing)
print("Resultados en train:")
lin_reg, lin_rmse, lin_mae = regresionLinear(listing_prepared, listing_label)
print("LinearRegression:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(lin_rmse, lin_mae))
tree_reg, tree_rmse, tree_mae = arbolDecision(listing_prepared, listing_label)
print("DecisionTreeRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(tree_rmse, tree_mae))
forest_reg, forest_rmse, forest_mae = bosqueAleatorio(listing_prepared, listing_label)
print("RandomForestRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(forest_rmse, forest_mae))
svm_reg, svr_rmse, svr_mae = kernelSVM(listing_prepared, listing_label)
print("SVR:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(svr_rmse, svr_mae))

# Validation
listing_v = pd.read_csv('validation.csv', sep=';', decimal='.')
listing_v = dataSetPreparation(listing)
listing_prepared_v, listing_label_v = dataSetPipelineTransform(listing_v)
print("Resultados en Validation:")

lin_predictions = lin_reg.predict(listing_prepared_v)
lin_mse = mean_squared_error(listing_label_v, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(listing_label_v, lin_predictions)
print("LinearRegression:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(lin_rmse, lin_mae))

tree_predictions = tree_reg.predict(listing_prepared_v)
tree_mse = mean_squared_error(listing_label_v, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_mae = mean_absolute_error(listing_label_v, tree_predictions)
print("DecisionTreeRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(tree_rmse, tree_mae))

forest_predictions = forest_reg.predict(listing_prepared_v)
forest_mse = mean_squared_error(listing_label_v, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_mae = mean_absolute_error(listing_label_v, forest_predictions)
print("RandomForestRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(forest_rmse, forest_mae))

svr_predictions = svm_reg.predict(listing_prepared_v)
svr_mse = mean_squared_error(listing_label_v, svr_predictions)
svr_rmse = np.sqrt(svr_mse)
svr_mae = mean_absolute_error(listing_label_v, svr_predictions)
print("SVR:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(svr_rmse, svr_mae))

# Test
listing_t = pd.read_csv('test.csv', sep=';', decimal='.')
listing_t = dataSetPreparation(listing)
listing_prepared_t, listing_label_t = dataSetPipelineTransform(listing_t)
print("Resultados en Validation:")

lin_predictions = lin_reg.predict(listing_prepared_v)
lin_mse = mean_squared_error(listing_label_v, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(listing_label_v, lin_predictions)
print("LinearRegression:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(lin_rmse, lin_mae))

tree_predictions = tree_reg.predict(listing_prepared_v)
tree_mse = mean_squared_error(listing_label_v, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_mae = mean_absolute_error(listing_label_v, tree_predictions)
print("DecisionTreeRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(tree_rmse, tree_mae))

forest_predictions = forest_reg.predict(listing_prepared_v)
forest_mse = mean_squared_error(listing_label_v, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_mae = mean_absolute_error(listing_label_v, forest_predictions)
print("RandomForestRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(forest_rmse, forest_mae))

svr_predictions = svm_reg.predict(listing_prepared_v)
svr_mse = mean_squared_error(listing_label_v, svr_predictions)
svr_rmse = np.sqrt(svr_mse)
svr_mae = mean_absolute_error(listing_label_v, svr_predictions)
print("SVR:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(svr_rmse, svr_mae))

import joblib
joblib.dump(lin_reg, "lin_reg.pkl")
joblib.dump(tree_reg, "tree_reg.pkl")
joblib.dump(forest_reg, "forest_reg.pkl")
joblib.dump(svm_reg, "svm_reg.pkl")
