# PREPROCESAMIENTO DE DATOS
import pandas as pd
from sklearn.preprocessing import StandardScaler

def cargar_y_preprocesar_datos(ruta_archivo):
    # Cargar el conjunto de datos
    data = pd.read_csv(ruta_archivo)

    # Normalizar las características numéricas
    numeric_features = data.select_dtypes(include=['float64']).columns
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data
