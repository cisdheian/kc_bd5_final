from sklearn.model_selection import train_test_split


def dataSetSplit(df):
    """
    Funcion para separar un dataset en sus 3 componentes de train, validation y test y guardarlos en la carpeta local
    :param df: dataset a separar
    :return: None
    """
    train, test = train_test_split(df, test_size=0.20, shuffle=True, random_state=0)
    train, validation = train_test_split(train, test_size=0.10, shuffle=True, random_state=0)

    train.to_csv('train.csv', sep=';', decimal='.', index=False, mode='w+')
    validation.to_csv('validation.csv', sep=';', decimal='.', index=False, mode='w+')
    test.to_csv('test.csv', sep=';', decimal='.', index=False, mode='w+')

    del train, validation, test


initial_columns = ["host_since", "host_is_superhost", "host_listings_count", "host_identity_verified",
                   "number_of_reviews", "review_scores_rating", "neighbourhood", "neighbourhood_cleansed",
                   "neighbourhood_group_cleansed", "latitude", "longitude", "property_type", "room_type",
                   "accommodates", "bathrooms", "bathrooms_text", "bedrooms", "beds", "amenities", "price"]

from datetime import date
from Utilities import getFecha, getCleanPrice
from ConfigAPI import getGeoCode
import numpy as np

amenities_list = ['Bedroom comforts', 'Hot water', 'Kitchen', 'TV', 'Cable TV', 'Pocket wifi', 'Changing table',
                  'Patio or balcony', 'Waterfront', 'Bread maker', 'Full kitchen', 'Heating', 'Single level home',
                  'Bathroom essentials', 'Pets allowed', 'Game console', 'Cooking basics', 'Shampoo', 'Bed linens',
                  'First aid kit', 'Piano', 'Dishwasher', 'Extra pillows and blankets', 'Iron', 'Self check-in',
                  'Garden or backyard', 'Baby monitor', 'Beachfront', 'Fire extinguisher', 'Window guards',
                  'Barbecue utensils', 'Dishes and silverware', 'Ethernet connection', 'Coffee maker', 'BBQ grill',
                  'Lake access', 'Hot tub', 'Long term stays allowed', 'Laptop-friendly workspace',
                  'Paid parking on premises', 'Stove', 'Smoke alarm', 'Carbon monoxide alarm', 'Microwave',
                  'Refrigerator', 'Free parking on premises', 'Lockbox', 'Paid parking off premises', 'Ski-in/Ski-out',
                  'Shower gel', 'Breakfast', 'Dryer', 'Free street parking', 'Air conditioning', 'Stair gates', 'Wifi',
                  'Baking sheet', 'Lock on bedroom door', 'Host greets you', 'Suitable for events',
                  'Cleaning before checkout', 'Hair dryer', 'Children’s books and toys', 'Room-darkening shades',
                  'Outlet covers', 'Keypad', 'Gym', 'Fireplace guards', 'Indoor fireplace', 'Table corner guards',
                  'Smoking allowed', 'Children’s dinnerware', 'Essentials', 'Washer', 'Baby bath',
                  'Luggage dropoff allowed', 'Elevator', 'Smart lock', 'Babysitter recommendations', 'EV charger',
                  'Building staff', 'Private living room', 'Oven', 'Beach essentials', 'Crib', 'Private entrance',
                  'Pack ’n Play/travel crib', 'Pool', 'Bathtub', 'High chair', 'Hangers']

limite_inferior = 10
limite_superior = 200


def dataSetPreparation(df):
    for c in df.columns:
        if not c in initial_columns:
            df = df.drop([c], axis=1)
    end_year = date.today().year
    df["antiguedad_host"] = df.host_since.apply(lambda c: end_year - getFecha(c).year)
    df.host_is_superhost = df.host_is_superhost.apply(lambda c: 0 if c == "f" else 1)
    df.host_identity_verified = df.host_identity_verified.apply(lambda c: 0 if c == "f" else 1)
    df["price_clean"] = df.price.apply(lambda c: getCleanPrice(c))
    df = df.drop(["host_since", "price"], axis=1)
    df = df.loc[np.logical_and(df[variable_objetivo] >= 10, df[variable_objetivo] <= 200)]
    df["latlng"] = df.apply(lambda r: "{},{}".format(r.latitude, r.longitude), axis=1)
    df["api_result"] = df.latlng.apply(lambda x: getGeoCode(x))
    df["postal_code"] = df["api_result"].apply(
        lambda x: x.get("postal_code", "")[0] if len(x.get("postal_code", "")) > 0 else "")
    df = df.drop(
        ["neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude", "latlng",
         "api_result"], axis=1)
    df = df.drop(["bathrooms"], axis=1)
    for a in amenities_list:
        df[a] = df.amenities.loc[np.logical_not(df.amenities.isna())].apply(lambda c: 1 if a in c else 0)
    df = df.drop(["amenities"], axis=1)
    return df


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


def dataSetPipelineTransform(df, fit=False):
    for x in variables_categoricas:
        df[x] = df[x].astype(str)
    for x in variables_binarias:
        df[x] = df[x].fillna(0)
        df[x] = df[x].astype(np.int8)
    for x in variables_numericas:
        df[x] = df[x].apply(lambda c: getCleanPrice(c))
        df[x] = df[x].astype(float)
    if fit:
      return full_pipeline.fit_transform(df), df[variable_objetivo]
    else:
      return full_pipeline.transform(df), df[variable_objetivo]

