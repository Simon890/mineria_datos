### Importacion de modulos ###

# Procesamiento de datos
import pandas as pd
import numpy as np

# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Dimensionalidad
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

# Metricas
from gap_statistic import OptimalK
from sklearn.metrics import silhouette_score

# Otros
from IPython.display import display



### Dataset ###

data = pd.read_csv("Crop_recommendation.csv")
display(data.head())



### Tratamiento Primario de datos ###

# sobre el dataset
display(data.info())
display(data.describe())


#---- Analisis de valores nulos ----#

# Calcular la cantidad de valores nulos y no nulos para cada columna
null_counts = data.isnull().sum()
non_null_counts = data.notnull().sum()

total_records = len(data)
info_df = pd.DataFrame({'Variable': data.columns, 'Nulos': null_counts, 'No Nulos': non_null_counts})


# Configurar el gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Crear el gráfico de barras
info_df.plot(x='Variable', y='No Nulos', kind='bar', ax=ax, color='green', label='No Nulos')
info_df.plot(x='Variable', y='Nulos', kind='bar', ax=ax, color='red', label='Nulos', bottom=non_null_counts)

# Configurar la línea punteada horizontal
plt.axhline(y=total_records, color='gray', linestyle='--', label='Total Registros')

ax.set_xlabel('Variable')
ax.set_ylabel('Cantidad de Valores')
ax.set_title('Cantidad de Valores Nulos y No Nulos por Variable')
ax.set_ylim([0, 2500])

plt.legend()
plt.xticks(rotation=45, ha='right')

plt.tight_layout(), plt.show()


#--- Outliers ---#
sns.boxplot(data), plt.show()


#--- Estadisticos de forma grafica --#
nums = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = data['label']

# Configuración del estilo
sns.set(style="whitegrid")
colors = ['#6096B4', '#85586F', '#FD8A8A', '#98A8F8', '#D18CE0', '#6096B4', '#85586F', '#FD8A8A']

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10))

# Iterar a través de las variables y crear un gráfico para cada una
for idx, column in enumerate(nums.columns):
    ax = axes[idx // 2, idx % 2]  # Calcular la posición de la subtrama

    # Boxplot
    bp = ax.boxplot(nums[column], patch_artist=True, vert=False)
    bp['boxes'][0].set_facecolor(colors[idx])
    bp['boxes'][0].set_alpha(0.7)

    # Violin plot
    vp = ax.violinplot(nums[column], points=500, showmeans=False, showextrema=False, showmedians=False, vert=False)
    vp['bodies'][0].get_paths()[0].vertices[:, 1] = np.clip(vp['bodies'][0].get_paths()[0].vertices[:, 1], 0.8, 1.2)
    vp['bodies'][0].set_color(colors[idx])
    vp['bodies'][0].set_alpha(0.7)

    # Scatter plot
    y = np.full(len(nums[column]), 0.8) + np.random.uniform(low=-.1, high=.1, size=len(nums[column]))
    ax.scatter(nums[column], y, s=3, c=colors[idx])

    ax.set_title(column)
    ax.set_yticks([])
    ax.set_xlabel("Value")

# Añadir el gráfico de pastel en el último eje
ax_pie = axes[3, 1]
label_counts = labels.value_counts()
ax_pie.pie(label_counts, labels=label_counts.index, colors=sns.color_palette('Set3', len(label_counts)))
ax_pie.set_title("Label Distribution")

plt.tight_layout(), plt.show()


#--- Graficos para entender los outliers ---#

cols=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

for col in cols:
    plt.figure(figsize=(15,6))
    plt.title(col,fontsize=20)
    
    df = data.groupby('label').agg({col:np.median}).reset_index().sort_values([col])
    g = sns.boxplot(x=data['label'],y=data[col],color="#426579",data=data,order = df.label.values)
    g.tick_params(axis='x', labelrotation=90)
    g.tick_params(axis='y')
    
    plt.show()



### Estandarizacion de datos ###
standar = StandardScaler()
robust = RobustScaler()

# Escalar los datos de forma robusta (con menor efecto sobre atipicos)
data_standar = pd.DataFrame(standar.fit_transform(nums), columns=nums.columns)
data_scaled = pd.DataFrame(robust.fit_transform(data_standar), columns=data_standar.columns)

display('Original', data.describe()[1:3])
display('Escalado', data_scaled.describe()[1:3])


#--- Distribuciones de datos estandarizados ---#
sns.boxplot(data_scaled), plt.show()



### Correlaciones ###

# heatmap
sns.heatmap(data_scaled.corr(), annot=True), plt.show()

# distribuciones todos vs todos
sns.pairplot(data=data_scaled, height=2), plt.show()



### Analisis de Componentes Principales ###

#--- Obtencion de los componentes ---#
pca = PCA()

pca_result = pca.fit_transform(data_scaled)

pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i, _ in enumerate(data_scaled.columns)])
pca_df["label"] = data["label"]

display(pca_df.head())


#--- Explicabilidad de los componentes ---#

varianza = np.cumsum(pca.explained_variance_ratio_)

acum = pd.DataFrame({"Component": pca_df.columns[:-1],
                     "EigenValues": pca.explained_variance_,
                     "PropVar": pca.explained_variance_ratio_,
                     "VarAcum": varianza})
display(acum)

#--- grafico de explicabilidad ---#

plt.bar(range(1, 8), pca.explained_variance_ratio_, alpha=0.5, align="center")
plt.plot(range(1, 8), varianza, marker="o" ,color="red")
plt.ylabel("Proporción de varianza explicada")
plt.xlabel("Componentes principales")
plt.show()

#--- Seleccion de componentes ---#

pcas = pca_df[["PC1", "PC2", "PC3"]]
sns.heatmap(pcas.corr(), annot=True), plt.show()

#--- Visualizar la distribucion de las componentes por clase ---#
fig = px.scatter_3d(pca_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='label',
                    labels={"color":"label"})
fig.show()



### Escalado Multidimensional ###


#--- Definicion de ISOMAP method ---#

def isomap_vars(n_neighbors, n_components, comp1, comp2, comp3=None):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)

    isomap.fit(data_scaled)
    datos_isomap = isomap.transform(data_scaled)

    isomap_df = pd.DataFrame(datos_isomap, columns=[f"C{i + 1}" for i in range(n_components)])
    isomap_df["label"] = data["label"]

    if comp3 is None:
        grupos = isomap_df.groupby("label")

        for nombre, grupo in grupos:
            plt.title(f"ISOMAP - Neighs = {n_neighbors} - Comps = {n_components}")
            plt.xlabel(comp1), plt.ylabel(comp2)
            plt.plot(grupo[comp1], grupo[comp2], marker="o", linestyle="", markersize=5, label=nombre)
    else:
        return isomap_df


#--- visualizacion de componentes ---#

#--- Variacion de la cantidad de vecinos ---#

fig, axes = plt.subplots(ncols=3, figsize=(20, 6))

plt.sca(axes[0])
isomap_vars(3, 2, "C1", "C2")

plt.sca(axes[1])
isomap_vars(6, 2, "C1", "C2")

plt.sca(axes[2])
isomap_vars(12, 2, "C1", "C2")

plt.show()

#--- Variciacion usando tres componentes ---#

isomap_df = isomap_vars(12, 3, 'C1', 'C2', 'C3')
fig = px.scatter_3d(isomap_df,
                    x='C1',
                    y='C2',
                    z='C3',
                    color='label',
                    labels={"color":"label"})
fig.show()



### Algortimo TSNE ###

#--- Definicion del algoritmo ---#

def apply_tsne(data, n_components, perplexity, n_iter):
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=perplexity, n_iter=n_iter)
    return tsne.fit_transform(data)

def plot_tsne(results, title, ax):
    sns.scatterplot(
        x=results[:, 0], y=results[:, 1],
        hue=data.label,
        palette=sns.color_palette("hls", 22),
        alpha=0.9, legend=False, ax=ax)

    ax.set_title(title)

def plot_tsne_3d(results):
    fig = px.scatter_3d(
        x=results[:, 0],
        y=results[:, 1],
        z=results[:, 2],
        color=data.label,
        labels={"color":"label"})

    fig.update_layout(title=f"t-SNE con n_components={3}")


#--- Prueba con diferentes valores de perplexity ---#
fig, axes = plt.subplots(ncols=3, figsize=(20, 6))

for i, perplexity in enumerate([5, 25, 50]):
    results = apply_tsne(data_scaled, 2, perplexity, 300)
    plot_tsne(results, f"t-SNE con perplexity={perplexity}", axes[i])

plt.show()


#--- Prueba con diferentes valores de n_iter ---#
fig, axes = plt.subplots(ncols=3, figsize=(20, 6))

for i, n_iter in enumerate([250, 500, 1000]):
    results = apply_tsne(data_scaled, 2, 30, n_iter)
    plot_tsne(results, f"t-SNE con n_iter={n_iter}", axes[i])

plt.show()


#--- Prueba con 3 n_components ---#

results = apply_tsne(data_scaled, 3, 30, 500)
fig = px.scatter_3d(x=results[:, 0],
                    y=results[:, 1],
                    z=results[:, 2],
                    color=data.label,
                    labels={"color":"label"})
fig.show()



### Clustering K-Means ###

#--- Scores ---#

ranges = range(1, data["label"].unique().shape[0])
kmeans_arr = [KMeans(n_clusters=i) for i in ranges]
scores_arr = [(kmeans.fit(data_scaled).score(data_scaled)*(-1)) for kmeans in kmeans_arr]

sns.lineplot(scores_arr)
plt.xticks(ranges)
plt.show()


#--- Seleccion de 6 clusters ---#

kmeans_model = KMeans(6)
kmeans_model.fit(data_scaled)

kmeans_data = data_scaled.copy()
kmeans_data['tag'] = kmeans_model.predict(kmeans_data)

#--- Visualizar la distribucion de las componentes por clase ---#
fig = px.scatter_3d(kmeans_data,
                    x='temperature',
                    y='rainfall',
                    z='humidity',
                    color='tag',
                    labels={"color":"tag"},
                    color_discrete_sequence=px.colors.sequential.Plasma
                    )
fig.show()


#--- Encontrar el numero optimo de clusteres ---#
optimalk = OptimalK(n_iter=50)

optimalk(data_scaled, n_refs=200, cluster_array=range(1, data["label"].unique().shape[0]))


#--- Graficar los 20 clusteres obtenidos ---#
kmeans_model = KMeans(20)
kmeans_model.fit(data_scaled)

kmeans_data = data_scaled.copy()
centroids = kmeans_model.cluster_centers_
kmeans_data['tag'] = kmeans_model.predict(kmeans_data)


#--- Visualizar la distribucion de las componentes por clase ---#
fig = px.scatter_3d(kmeans_data,
                    x='temperature',
                    y='rainfall',
                    z='humidity',
                    color='tag',
                    labels={"color":"tag"},
                    )
fig.show()


#--- Obtencion de clusters para dataset reducido por PCA ---#
optimalk = OptimalK(n_iter=50)

optimalk(pca_df[['PC1', 'PC2', 'PC3']], n_refs=250, cluster_array=range(1, pca_df["label"].unique().shape[0]))

#--- Entrenar con el numero encontrado ---#
kmeans_model = KMeans(16)
kmeans_model.fit(pca_df[['PC1', 'PC2', 'PC3']])

kmeans_data = pca_df[['PC1', 'PC2', 'PC3']].copy()
centroids = kmeans_model.cluster_centers_
kmeans_data['tag'] = kmeans_model.predict(kmeans_data)


#--- Visualizar la distribucion de las componentes por clase ---#
fig = px.scatter_3d(kmeans_data,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='tag',
                    labels={"color":"tag"},
                    )
fig.show()



### Clustering usando algoritmos jerarquicos ###

#--- Comparacion de algoritmos divisorios ---#

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Calcular el clustering jerárquico con el método de Ward
cluster_ward = sch.linkage(data_scaled, method="ward", metric='euclidean')
dendrogram_ward = sch.dendrogram(cluster_ward, ax=axs[0])
axs[0].set_title("Dendrograma (método de Ward)")
axs[0].set_xlabel("Categorías")
axs[0].set_ylabel("Distancias Euclidianas")

# Calcular el clustering jerárquico con el método del centroide
cluster_centroid = sch.linkage(data_scaled, method="centroid", metric='euclidean')
dendrogram_centroid = sch.dendrogram(cluster_centroid, ax=axs[1])
axs[1].set_title("Dendrograma (método del centroide)")
axs[1].set_xlabel("Categorías")
axs[1].set_ylabel("Distancias Euclidianas")

# Mostrar la figura
plt.tight_layout()
plt.show()


#--- Score de Siluethe por numero de clusters ---#

# Define el rango de números de clusters que deseas probar, excluyendo 1
ranges_divisive = range(2, 22)
s_scores_divisive = []

for i in ranges_divisive:
    cluster_ward_divisive = sch.linkage(data_scaled, method="ward", metric='euclidean')
    labels_divisive = fcluster(cluster_ward_divisive, i, criterion='maxclust')
    
    # Calcula el score de silueta para este número de clusters
    s_score_divisive = silhouette_score(data_scaled, labels_divisive)
    s_scores_divisive.append(s_score_divisive)


s_scores_divisive_df = pd.DataFrame(s_scores_divisive, columns=["s_score"], index=ranges_divisive)

# Gráfico de Silhouette Score vs Número de Clusters para el algoritmo divisivo
plt.figure(figsize=(10, 5))
sns.lineplot(data=s_scores_divisive_df, x=s_scores_divisive_df.index, y="s_score", marker="o", markerfacecolor="#ff6961")
plt.title("Silhouette Scores vs N Clusters for Divisive Hierarchical Clustering")
plt.xlabel("Número de Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(ranges_divisive)
plt.grid(True)
plt.show()


#--- Cortar el dendrograma para obtener 14 clusters ---#

labels = fcluster(cluster_ward, 14, criterion='maxclust')

dendo_data = pca_df.copy()
dendo_data["tag"] = labels

# Crear un dendrograma con etiquetas para 14 clusters
dendrogram_14_clusters = sch.dendrogram(cluster_ward, p=14, truncate_mode='lastp', labels=labels)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.title('Dendrograma con 14 Clusters')
plt.show()


#--- Algoritmos Jerarquicos aglormerativos ---#

#--- Obtener score de siluethe para cada cantidad de clusteres ---#

ranges = range(2, data["label"].unique().shape[0])
ac_arr = [AgglomerativeClustering(n_clusters=i, metric="euclidean", linkage="ward") for i in ranges]
y_ac_arr = [ac.fit_predict(data_scaled) for ac in ac_arr]

s_scores = [silhouette_score(data_scaled, y_ac) for y_ac in y_ac_arr]
s_scores_df = pd.DataFrame(s_scores, columns=["s_score"], index=ranges)

sns.lineplot(s_scores_df, x=s_scores_df.index, y="s_score", marker="o", markerfacecolor="#ff6961").set(title="Silhouette Scores vs N Clusters for Aglomerative", xlabel="N Clusters", ylabel="Slihouette Score")
plt.xticks(ranges)
plt.show()


#--- Agrupamiento aglomerativo del dataset reducido por pca ---#

aglo_data = pca_df[['PC1', 'PC2', 'PC3']].copy()

aglo_model = AgglomerativeClustering(n_clusters=16, metric="euclidean", linkage="ward")
tags = aglo_model.fit_predict(aglo_data)

print(f"score de silhouette para dataset reducido por pca y agrupado por aglomeramiento: {silhouette_score(aglo_data, tags)}")

aglo_data['tag'] = tags

colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
           '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']

aglo_data['color'] = [colores[tag] for tag in tags]

fig = px.scatter_3d(aglo_data,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='color',
                    labels={"color":"tag"},
                    )
fig.show()

