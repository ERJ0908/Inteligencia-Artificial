# Visualización de Resultados de Clasificación
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def graficar_matrices_de_confusion(y_test, predicciones, etiquetas):
    for pred, etiqueta in zip(predicciones, etiquetas):
        plt.figure(figsize=(10, 6))
        sns.heatmap(pd.DataFrame(confusion_matrix(y_test, pred)), annot=True, cmap="YlGnBu", fmt='g')
        plt.title(f'Matriz de Confusión - {etiqueta}')
        plt.show()

def graficar_dispersion_adicional(data):
    # Generar gráficos de dispersión adicionales
    sns.pairplot(data, hue='Class', palette='viridis')
    plt.suptitle('Pairplot de Características por Clase', y=1.02)
    plt.show()

    sns.pairplot(data, hue='Cluster', palette='viridis')
    plt.suptitle('Pairplot de Características por Cluster', y=1.02)
    plt.show()
