"""
Archivo de configuracion para el consumo de las API
"""
import requests
from urllib import parse
import json
import pandas as pd

# Geocoding API KEY
GAPIK = "AIzaSyCVHVA98CtG7PwsvY0Igd4HmlzDk1hz5_M"

Geo_Config = {
    "url": "https://maps.googleapis.com/maps/api/geocode/json"
    , "key": GAPIK
}


def getGeoCode(latlng):
    """
    Funcion para buscar la informacion geografica a partir de latitude y longitud
    :param latlng: Variable combinada que contiene latitude y longitud separados por coma
    :return: diccionario con informacion relevante del punto geografico: Global code, postal code, neighborhood
    """
    url = Geo_Config["url"]
    key = Geo_Config["key"]
    parameter_dict = {'latlng': latlng, 'key': key}
    parameters = parse.urlencode(parameter_dict)
    v_request = url + "?" + parameters
    v_response = requests.get(v_request)
    v_result = v_response.content.decode()
    v_data_result = json.loads(v_result)
    gcode = v_data_result.get("plus_code", {'compound_code': '', 'global_code': ''}).get('global_code')
    components = v_data_result.get("results", [{"address_components": []}])
    df_aux = pd.DataFrame()
    pcode = []
    ncode = []
    for x in components:
        for y in x.get("address_components"):
            if "postal_code" in y.get("types", []):
                pcode.append(y.get("short_name"))
            if "neighborhood" in y.get("types", []):
                ncode.append(y.get("short_name"))
    return {"global_code": gcode, "postal_code": list(set(pcode)), "neighborhood": list(set(ncode))}
