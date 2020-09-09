# Archivo con funciones varias
import wget
import gzip
import pandas as pd

RUTA_DS = "http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2020-08-25/data/"


def getDF(gzip_path, csv_path):
    """
    Función para descargar el dataset y cargarlo a una DataFrame de Pandas
    gzip_path: Ruta del archivo gzip
    csv_path: Rutal del archivo descomprimido
    """
    url = RUTA_DS + gzip_path
    filename = wget.download(url)
    with gzip.open(gzip_path, 'rb') as gz:
        file_content = gz.read()
        with open(csv_path, "w+") as csv:
            csv.write(file_content.decode("utf-8"))
    return pd.read_csv(csv_path)


from datetime import date


def getFecha(o):
    """
    Funcion para convertir de un object/string a fecha
    :param o: variable object a transformar
    :return: fecha transformada o su valor default '1970-1-1'
    """
    start_date = date(1970, 1, 1)
    fecha = start_date
    try:
        so = o.strip("").split("-")
        fecha = date(int(so[0]), int(so[1]), int(so[2]))
    except:
        fecha = start_date
    return fecha


import numpy as np
import re


def getCleanPrice(o):
    remove = re.compile(r'[^\d\.]+')
    s = "0.00"
    try:
        s = remove.sub(" ", str(o)).strip()
    except:
        s = "0.00"
    f = 0.00
    try:
        f = np.float64(s)
    except:
        f = np.float64(0.00)
    return f



