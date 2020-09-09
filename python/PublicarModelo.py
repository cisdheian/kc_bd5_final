import warnings
warnings.filterwarnings('ignore')
from tabpy_client import Client

import joblib
#Cargamos los modelos preentrenados
tree_reg = joblib.load("tree_reg.pkl")
forest_reg = joblib.load( "forest_reg.pkl")

variables_categoricas = ["property_type", "room_type", "bathrooms_text", "postal_code"]
variables_binarias = ["host_is_superhost", "host_identity_verified", "Bedroom comforts", "Hot water", "Kitchen", "TV",
                      "Cable TV", "Pocket wifi", "Changing table", "Patio or balcony", "Waterfront", "Bread maker",
                      "Full kitchen", "Heating", "Single level home", "Bathroom essentials", "Pets allowed",
                      "Game console", "Cooking basics", "Shampoo", "Bed linens", "First aid kit", "Piano", "Dishwasher",
                      "Extra pillows and blankets", "Iron", "Self check-in", "Garden or backyard", "Baby monitor",
                      "Beachfront", "Fire extinguisher", "Window guards", "Barbecue utensils", "Dishes and silverware",
                      "Ethernet connection", "Coffee maker", "BBQ grill", "Lake access", "Hot tub",
                      "Long term stays allowed", "Laptop-friendly workspace", "Paid parking on premises", "Stove",
                      "Smoke alarm", "Carbon monoxide alarm", "Microwave", "Refrigerator", "Free parking on premises",
                      "Lockbox", "Paid parking off premises", "Ski-in/Ski-out", "Shower gel", "Breakfast", "Dryer",
                      "Free street parking", "Air conditioning", "Stair gates", "Wifi", "Baking sheet",
                      "Lock on bedroom door", "Host greets you", "Suitable for events", "Cleaning before checkout",
                      "Hair dryer", "Children’s books and toys", "Room-darkening shades", "Outlet covers", "Keypad",
                      "Gym", "Fireplace guards", "Indoor fireplace", "Table corner guards", "Smoking allowed",
                      "Children’s dinnerware", "Essentials", "Washer", "Baby bath", "Luggage dropoff allowed",
                      "Elevator", "Smart lock", "Babysitter recommendations", "EV charger", "Building staff",
                      "Private living room", "Oven", "Beach essentials", "Crib", "Private entrance",
                      "Pack ’n Play/travel crib", "Pool", "Bathtub", "High chair", "Hangers"]
variables_numericas = ["host_listings_count", "number_of_reviews", "review_scores_rating", "accommodates", "bedrooms",
                       "beds", "antiguedad_host"]
variable_objetivo = "price_clean"

import numpy as np
import re

def getCleanValue(o):
    remove = re.compile(r'[^\d\.]+')
    s = "0.00"
    try:
        s = remove.sub(" ", str(o)).strip()
    except:
        s = "0.00"
    f=0.00
    try:
        f= np.float64(s)
    except:
        f = np.float64(0.00)
    return f
	
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=True)),
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=0.00, strategy="mean")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, variables_numericas),
    ("cat", cat_pipeline, variables_categoricas),
    ("bool", StandardScaler(), variables_binarias)
])


#Importamos un dataset ya preparado para hacer fit de los pipelines de tranformacion a utilizar
import pandas as pd

df = pd.read_csv("train_prepared.csv", index_col=0)
for x in variables_categoricas:
    df[x] = df[x].astype(str)
for x in variables_binarias:
    df[x] = df[x].fillna(0)
    df[x] = df[x].astype(np.int8)
for x in variables_numericas:
    df[x] = df[x].apply(lambda c: getCleanValue(c))
    df[x] = df[x].astype(float)
full_pipeline.fit_transform(df)

def hostPricePredictor(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11, _arg12, _arg13, _arg14):
    #Transformamos a diccionario para cargar en un DataFrame
    tab_datos = {"host_is_superhost": _arg1
                ,"host_listings_count": _arg2
                ,"host_identity_verified": _arg3
                ,"property_type": _arg4
                ,"room_type": _arg5
                ,"accommodates": _arg6
                ,"bathrooms_text": _arg7
                ,"bedrooms": _arg8
                ,"beds": _arg9
                ,"number_of_reviews": _arg10
                ,"review_scores_rating": _arg11
                ,"antiguedad_host": _arg12
                ,"postal_code": _arg13
                , "amenities": _arg14
                }

    df = pd.DataFrame(tab_datos)
    
    for a in amenities_list:
        df[a] = df.amenities.loc[np.logical_not(df.amenities.isna())].apply(lambda c: 1 if a in c else 0)
    df = df.drop(["amenities"], axis=1)
    
    for x in variables_categoricas:
        df[x] = df[x].astype(str)
    for x in variables_binarias:
        df[x] = df[x].fillna(0)
        df[x] = df[x].astype(np.int8)
    for x in variables_numericas:
        df[x] = df[x].apply(lambda c: getCleanPrice(c))
        df[x] = df[x].astype(float)
    full_pipeline.transform(df)
    tree_predictions = tree_reg.predict(listing_prepared)
    tree_mse = mean_squared_error(listing_label, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(listing_label, tree_predictions)
    print("DecisionTreeRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(tree_rmse, tree_mae))
    print("-"*100)
    forest_predictions = forest_reg.predict(listing_prepared)
    forest_mse = mean_squared_error(listing_label, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = mean_absolute_error(listing_label, forest_predictions)
    print("RandomForestRegressor:\nRMSE = ${:.4f}\nMAE = ${:.4f}".format(forest_rmse, forest_mae))
    
    return [price for price in forest_predictions]



cliente = Client("http://localhost:9004")
cliente.deploy("hostPricePredictor", hostPricePredictor, "Calcula el precio del hosting segun los parametros recibidos", override = True)	