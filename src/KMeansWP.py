import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


def camins_update(ncenters, ninit, maxiter, u, v, hgt, algo, scaler_type='none', joint_scaling=True):
    """
    Executa o algoritmo K-Means para clusterização de dados atmosféricos, retornando os centros dos clusters e outras estatísticas relacionadas.
    
    Parâmetros:
    - ncenters (int): Número de clusters desejados no K-Means.
    - ninit (int): Número de inicializações para o algoritmo K-Means.
    - maxiter (int): Número máximo de iterações para uma única execução.
    - u (xarray Dataset): Dados de reanálise da componente zonal do vento.
    - v (xarray Dataset): Dados de reanálise da componente meridional do vento.
    - hgt (xarray Dataset): Dados de reanálise da altura geopotencial.
    - algo (str): Nome do algoritmo a ser utilizado pelo K-Means.
    - scaler_type (str): Tipo de normalização a ser aplicada ('standard' para StandardScaler, 'minmax' para MinMaxScaler ou 'none' para não aplicar).
    - joint_scaling (bool): Se True, aplica a normalização em conjunto para todos os dados; se False, normaliza separadamente.

    Retorna:
    - u_wt (numpy array): Centros dos clusters para a componente zonal do vento.
    - v_wt (numpy array): Centros dos clusters para a componente meridional do vento.
    - hgt_wt (numpy array): Centros dos clusters para a altura geopotencial.
    - cf (numpy array): Fração de cada cluster no total de dados.
    - lat (numpy array): Vetor de latitudes.
    - lon (numpy array): Vetor de longitudes.
    - mk (KMeans object): Objeto KMeans após treinamento.
    - tempo (numpy array): Vetor de tempo.
    
    Notas:
    - O pré-processamento inclui a transformação dos conjuntos de dados em matrizes 2D onde cada linha representa um ponto no tempo e cada coluna um ponto no espaço (latitude x longitude).
    - A normalização é opcional e pode ser feita em todos os campos conjuntamente ou individualmente, dependendo do argumento 'joint_scaling'.
    - Os dados de entrada devem ser xarray Datasets com dimensões coerentes entre eles.
    - A função também calcula a fração de cada cluster no total de dados, que pode ser útil para análises estatísticas subsequentes.

    """
    
    lat = u['latitude'].squeeze()
    lon = u['longitude'].squeeze()
    tempo = u['time'].squeeze()

    dsk_hgt2 = hgt.values
    dsk_u2 = u.values
    dsk_v2 = v.values
    nt,ny,nx = dsk_hgt2.shape
    dsk_hgt = np.reshape(dsk_hgt2, [nt, ny*nx], order='F')
    dsk_u = np.reshape(dsk_u2, [nt, ny*nx], order='F')
    dsk_v = np.reshape(dsk_v2, [nt, ny*nx], order='F')
    
    if scaler_type == 'standard':
        Scaler = StandardScaler
    elif scaler_type == 'minmax':
        Scaler = MinMaxScaler
    else:
        Scaler = None
    
    if joint_scaling:
        dsk_means = np.concatenate((dsk_u, dsk_v, dsk_hgt), axis=1)
        if Scaler:
            scaler = Scaler()
            scaled = scaler.fit(dsk_means)
            scaled_dskmeans = scaled.transform(dsk_means)
        else:
            scaled_dskmeans = dsk_means
    else:
        if Scaler:
            scaler_u = Scaler()
            scaled_u = scaler_u.fit(dsk_u)
            dsk_u_scaled = scaled_u.transform(dsk_u)
            
            scaler_v = Scaler()
            scaled_v = scaler_v.fit(dsk_v)
            dsk_v_scaled = scaled_v.transform(dsk_v)
            
            scaler_hgt = Scaler()
            scaled_hgt = scaler_hgt.fit(dsk_hgt)
            dsk_hgt_scaled = scaled_hgt.transform(dsk_hgt)
        else:
            dsk_u_scaled = dsk_u
            dsk_v_scaled = dsk_v
            dsk_hgt_scaled = dsk_hgt
        
        #dsk_u = np.reshape(dsk_u, [-1, len(lat)*len(lon)], order='F')
        #dsk_v = np.reshape(dsk_v, [-1, len(lat)*len(lon)], order='F')
        #dsk_hgt = np.reshape(dsk_hgt, [-1, len(lat)*len(lon)], order='F')
        
        scaled_dskmeans = np.concatenate((dsk_u_scaled, dsk_v_scaled, dsk_hgt_scaled), axis=1)
    
    mk = KMeans(n_clusters=ncenters, algorithm=algo).fit(scaled_dskmeans)
    
    slcenter = len(lat) * len(lon)
    centers = mk.cluster_centers_
    
    # Inverse transform on cluster centers
    if joint_scaling and Scaler:
        centers = scaled.inverse_transform(centers)
    elif not joint_scaling and Scaler:
        centers[:, 0:slcenter] = scaled_u.inverse_transform(centers[:, 0:slcenter])
        centers[:, slcenter:slcenter*2] = scaled_v.inverse_transform(centers[:, slcenter:slcenter*2])
        centers[:, slcenter*2:slcenter*3] = scaled_hgt.inverse_transform(centers[:, slcenter*2:slcenter*3])
    
    centers_u = centers[:, 0:slcenter]
    centers_v = centers[:, slcenter:slcenter*2]
    centers_hgt = centers[:, slcenter*2:slcenter*3]
    
    u_wt = np.zeros((ncenters, len(lat), len(lon)))
    v_wt = np.zeros((ncenters, len(lat), len(lon)))
    hgt_wt = np.zeros((ncenters, len(lat), len(lon)))
    cf = np.zeros((ncenters))
    
    def get_cluster_fraction(m, label):
        kf = (m.labels_ == label).sum() / (m.labels_.size * 1.0)
        return kf * 100 
    
    for i in range(mk.n_clusters):    
        u_wt[i] = centers_u[i, :].reshape(len(lat), len(lon), order='F')
        v_wt[i] = centers_v[i, :].reshape(len(lat), len(lon), order='F')
        hgt_wt[i] = centers_hgt[i, :].reshape(len(lat), len(lon), order='F')
        cf[i] = get_cluster_fraction(mk, i)
    
    return u_wt, v_wt, hgt_wt, cf, lat, lon, mk, tempo