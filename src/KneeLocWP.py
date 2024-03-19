import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import os 
import pathlib



def prekneed1(u,v,hgt):
    #u = u.to_array().squeeze()
    #v = v.to_array().squeeze()
    #hgt = hgt.to_array().squeeze()

    dsk_hgt2 = hgt.values
    dsk_u2 = u.values
    dsk_v2 = v.values
    nt,ny,nx = dsk_hgt2.shape
    dsk_hgt = np.reshape(dsk_hgt2, [nt, ny*nx], order='F')
    dsk_u = np.reshape(dsk_u2, [nt, ny*nx], order='F')
    dsk_v = np.reshape(dsk_v2, [nt, ny*nx], order='F')
    dsk_means = np.concatenate((dsk_u,dsk_v,dsk_hgt), axis=1)
    #scaler = StandardScaler()
    #scaled_dskmeans = scaler.fit_transform(dsk_means)
    scaled_dskmeans = dsk_means
    return scaled_dskmeans





def teste_kmeans(q, scaled_dskmeans):
    sse = []
    for k in range(1, q):
        kmeans_kwargs = {   "init": "random",
                    "n_init": 10,
                    "max_iter": 300,
                    "random_state": 42,
                }
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_dskmeans)
        sse.append(kmeans.inertia_)
    
    return sse


def plot_knee(q,sse,PDIR):
        
    kl = KneeLocator(range(1, q), sse, curve="convex", direction="decreasing")

        



    fig, ax = plt.subplots(1,1,figsize=(8, 6))
    ax.plot(range(1, q), sse, '-o', markersize=8, linewidth=2, label='SSE', color='#A1456D')
    ax.plot(kl.elbow, sse[kl.elbow - 1], marker='o', markersize=14, label=f'Ideal Clusters number = {kl.elbow}', color='red')
    #ax.annotate(f'Ideal Clusters number = {kl.elbow}', xy=(kl.elbow, sse[kl.elbow - 2]), xytext=(kl.elbow-2, sse[kl.elbow - 1]-2), arrowprops=dict(facecolor='black', shrink=0.05),)
    
    
    # Configurando rótulos e título
    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('SSE', fontsize=14)
    ax.set_title('Elbow Method', fontsize=16)


    # Adicionando grade
    ax.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.5)
    #ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # Definindo cor de fundo
    #ax.set_facecolor('#FFF8D4')

    # Personalizando ticks e limites do eixo
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    # Adicionando a legenda e escolhendo sua posição
    ax.legend(fontsize=12, loc='upper right')


    # Configurando diretorio de salvamento
    PDIR2 = PDIR+'/figures/'
    file = 'elbow_idealclusters.png'

    pasta = pathlib.Path(PDIR2)

    pasta.mkdir(parents=True, exist_ok=True)


    # Salvando a figura
    plt.savefig(PDIR2+file, dpi=300)
    plt.close()
    return kl.elbow
    



    
