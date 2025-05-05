En el algoritmo K-Means, uno de los desafíos principales es determinar el número óptimo de clústeres (k) para un conjunto de datos dado. 
Existen varias técnicas propuestas en la literatura para abordar este problema. Implementaré y explicaré tres métodos populares: 
el método del codo (Elbow Method), el índice de silueta (Silhouette Score), y el criterio de Calinski-Harabasz.
Método del Codo (Elbow Method)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# Método del codo
def elbow_method(X, max_k=10):
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k+1), wcss, marker='o', linestyle='--')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('WCSS')
    plt.title('Método del Codo')
    plt.show()

elbow_method(X)

Índice de Silueta (Silhouette Score)
def silhouette_method(X, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Puntuación de Silueta')
    plt.title('Método de la Silueta')
    plt.show()

silhouette_method(X)

Criterio de Calinski-Harabasz
def calinski_harabasz_method(X, max_k=10):
    ch_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        score = calinski_harabasz_score(X, labels)
        ch_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k+1), ch_scores, marker='o', linestyle='--')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Puntuación Calinski-Harabasz')
    plt.title('Método Calinski-Harabasz')
    plt.show()

calinski_harabasz_method(X)

Implementación Integrada
def estimate_optimal_k(X, max_k=10):
    # Inicializar listas para almacenar resultados
    wcss = []
    silhouette_scores = []
    ch_scores = []
    
    # Calcular métricas para diferentes valores de k
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        if k >= 2:
            labels = kmeans.predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
            ch_scores.append(calinski_harabasz_score(X, labels))
    
    # Graficar resultados
    plt.figure(figsize=(15, 10))
    
    # Método del codo
    plt.subplot(3, 1, 1)
    plt.plot(range(1, max_k+1), wcss, marker='o', linestyle='--')
    plt.title('Método del Codo')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('WCSS')
    
    # Método de silueta
    plt.subplot(3, 1, 2)
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o', linestyle='--')
    plt.title('Método de la Silueta')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Puntuación de Silueta')
    
    # Método Calinski-Harabasz
    plt.subplot(3, 1, 3)
    plt.plot(range(2, max_k+1), ch_scores, marker='o', linestyle='--')
    plt.title('Método Calinski-Harabasz')
    plt.xlabel('Número de clústeres (k)')
    plt.ylabel('Puntuación CH')
    
    plt.tight_layout()
    plt.show()
    
    # Sugerir el mejor k basado en cada método
    optimal_k_elbow = np.argmin(np.diff(wcss)) + 2  # +2 porque diff reduce en 1 y k comienza en 2
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2
    optimal_k_ch = np.argmax(ch_scores) + 2
    
    print(f"Sugerencia de k óptimo:")
    print(f"- Método del codo: {optimal_k_elbow}")
    print(f"- Método de silueta: {optimal_k_silhouette}")
    print(f"- Método Calinski-Harabasz: {optimal_k_ch}")

# Ejemplo de uso
estimate_optimal_k(X)

