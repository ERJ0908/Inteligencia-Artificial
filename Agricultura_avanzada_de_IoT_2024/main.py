from preprocesamiento import cargar_y_preprocesar_datos
from clasificacion import dividir_datos, entrenar_clasificadores, evaluar_clasificadores
from clustering import realizar_clustering
from visualizacion import graficar_matrices_de_confusion, graficar_dispersion_adicional

def main():
    ruta_archivo = 'Advanced_IoT_Dataset.csv'
    
    # Preprocesamiento de datos
    data = cargar_y_preprocesar_datos(ruta_archivo)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = dividir_datos(data)
    
    # Entrenar los modelos de clasificación
    modelos = entrenar_clasificadores(X_train, y_train)
    
    # Evaluar los modelos de clasificación
    predicciones = evaluar_clasificadores(modelos, X_test, y_test)
    
    # Etiquetas de los modelos
    etiquetas = ['Regresión Logística', 'Random Forest', 'SVM']
    
    # Graficar matrices de confusión
    graficar_matrices_de_confusion(y_test, predicciones, etiquetas)
    
    # Realizar clustering y graficar resultados
    realizar_clustering(X_train, data)
    
    # Generar gráficos de dispersión adicionales
    graficar_dispersion_adicional(data)

if __name__ == "__main__":
    main()
