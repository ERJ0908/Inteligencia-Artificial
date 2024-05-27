
# REALIZANDO CLUSTERING
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def realizar_clustering(X, data):
    # Aplicar el algoritmo de clustering (KMeans)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Agregar los clusters al conjunto de datos original
    data['Cluster'] = clusters

    # Visualizar los clusters resultantes
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=' Average  of chlorophyll in the plant (ACHP)', y=' Plant height rate (PHR)', hue='Cluster', palette='viridis')
    plt.title('Resultados de Clustering')
    plt.show()

    # Analizar los clusters
    cluster_centers = kmeans.cluster_centers_
    print("Centros de los Clusters:\n", cluster_centers)

    # Visualizar la distribución de clases dentro de cada cluster
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Class', hue='Cluster')
    plt.title('Distribución de Clases por Cluster')
    plt.show()
