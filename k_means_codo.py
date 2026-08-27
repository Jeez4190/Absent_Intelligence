'''
    A P R E N D I Z A J E    N O    S U P E R V I S A D O
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# --- 1. Cargar dataset Iris ---
df = pd.read_csv("fisheriris.csv")

print("Primeras filas del dataset:")
print(df.head())

# Seleccionar solo columnas numéricas (las 4 características de Iris)
X = df.select_dtypes(include=['float64', 'int64']).values  

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Método del codo ---
wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, wcss, 'bo-', markersize=8)
plt.xlabel("Número de clusters (k)")
plt.ylabel("WCSS (Inercia)")
plt.title("Método del Codo para elegir k")
plt.grid(True)
plt.show()

# --- 3. Encontrar el "codo" automáticamente ---
# Usamos la técnica de la distancia máxima a la línea entre el primer y último punto
p1, p2 = np.array([1, wcss[0]]), np.array([10, wcss[-1]])
distancias = []
for i in range(len(K)):
    p = np.array([K[i], wcss[i]])
    dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
    distancias.append(dist)

k_optimo = K[np.argmax(distancias)]
print(f"\n📌 El número ideal de clusters es: k = {k_optimo}")

# --- 4. Ajustar KMeans con k óptimo ---
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

df['Cluster'] = y_kmeans

print("\nDataFrame con cluster asignado:")
print(df.head())

# --- 5. Visualizar en 2D usando PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_kmeans, cmap='viridis', s=40)
plt.scatter(kmeans.cluster_centers_[:,0:1], kmeans.cluster_centers_[:,1:2],
            c='red', marker='X', s=200, label="Centroides (en PCA)")
plt.title(f"K-Means aplicado al dataset Iris (k={k_optimo})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()
