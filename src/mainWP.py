


import numpy as np 
import pandas as pd
import pathlib
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings 


from boia_domain_disserta import CreateMapBoia
from GetDays import pega_dias, pega_dias24, pega_dias48
from KneeLocWP import prekneed1, teste_kmeans, plot_knee
from ProcessWP import le_dados_onda, le_dados_era, get_point_lat_lon, sel_time_era, sel_time_onda, sel_point_lat_lon, sel_var_onda, sel_time_era_extreme, before_days
from PlotWP import plota_clusters_extreme, plota_clusters_extreme_wave_rose, plota_clusters_extreme_waves_before, plot_hs_serie, monthly_counts
from SaveWP import save_array, save_kmeans, save_dict, save_df_boia_waverys
from KMeansWP import camins_update


per = 95
percentil = per
boia = 'santos'
testenome = f'teste_per{per}_'+ str(boia)
# Diretorio para salvar dados e figuras #

examples_path = '../examples/'

PDIR=examples_path+ f'{testenome}/'
pasta = pathlib.Path(PDIR)
ind = 1 

while pasta.exists():
    PDIR=examples_path+f'{testenome}_{ind}/'
    pasta = pathlib.Path(PDIR)
    ind += 1 

pasta.mkdir(parents=True, exist_ok=False)

print('1 - Selecionando ponto no mapa')
coords = get_point_lat_lon(boia,PDIR)

#startwtera = '1993-01-01'
#endwtera = '2023-10-31'
startwtera = '1993-01-01'
endwtera = '2018-01-01'
startwtwav = '1993-01-01'
endwtwav = '2018-01-01'
#startwtwav = '1993-01-01'
#endwtwav = '2023-10-31'
print('2 - Lendo Dados de onda')
ds_onda= le_dados_onda()
print(f'3 - Selecionando dias de ondas para analise entre {startwtwav} e {endwtwav}')
ds_wave = sel_time_onda(ds_onda,startwtwav,endwtwav)
ds_wave['time'] = pd.DatetimeIndex(ds_wave['time'].values).normalize()
lat_boia = coords['lat']
lon_boia = coords['lon']
ds_wave_point = sel_point_lat_lon(ds_wave, lat_boia, lon_boia)

#############################################################

# Selecionando dias de ondas extremas #
print('4 - Selecionando dias de ondas extremas')

ds_hs_ponto = sel_var_onda(ds_wave_point,'VHM0')
df_hs_ponto = ds_hs_ponto['VHM0'].to_dataframe()
Fperc95 = np.percentile(df_hs_ponto['VHM0'].dropna(),percentil)
mtempo95 = df_hs_ponto['VHM0'][df_hs_ponto['VHM0'] >= Fperc95]
wave95 = mtempo95.index.strftime('%Y-%m-%d')   

print(f'Quantidade de dais que excedem o percentil {per}: {len(wave95)}')

print(f'Percentil {per} de Hs: {Fperc95}')

print(f'Salva .csv do dataframe de Hs e Waverys')

#save_df_boia_waverys(boia, df_hs_ponto, PDIR)

print('5 - Plotando serie de hs')

plot_hs_serie(ds_hs_ponto,Fperc95,percentil, PDIR)


print('6 - Selecionando variaveis de onda')

ds_hs = sel_var_onda(ds_wave,'VHM0')
ds_tp = sel_var_onda(ds_wave,'VTPK')
ds_dir = sel_var_onda(ds_wave, 'VMDR')


print('7 - Selecionando dias de ondas extremas no ponto')


# Lendo dado atmosferico #
print('8 - Lendo dado atmosferico')
ds_u1, ds_v1, ds_hgt1 = le_dados_era()
# Selecionando tempo para trabalhar com o dado atmosferico #
print(f'9 - Selecionando tempo para trabalhar com o dado atmosferico entre {startwtera} e {endwtera}')
ds_u, ds_v, ds_hgt = sel_time_era(ds_u1,ds_v1,ds_hgt1,startwtera,endwtera)
# Selecionando dias de onda extrema no dado atmosferico #
print('10 - Selecionando dias de onda extrema no dado atmosferico')
ds_u['time'] = pd.DatetimeIndex(ds_u['time'].values).normalize()
ds_v['time'] = pd.DatetimeIndex(ds_v['time'].values).normalize()
ds_hgt['time'] = pd.DatetimeIndex(ds_hgt['time'].values).normalize()


ds_u_sel, ds_v_sel, ds_hgt_sel = sel_time_era_extreme(ds_u, ds_v, ds_hgt, wave95)
#ds_u2_sel, ds_v2_sel, ds_hgt2_sel = sel_time_era_extreme(ds_u, ds_v, ds_hgt, wave95_minus2)



# Obtendo clusters ideais 
print('11 - Definindo clusters ideais')
ickmeans = prekneed1(ds_u_sel,ds_v_sel,ds_hgt_sel)
clmax = 21
sseclusters = teste_kmeans(clmax, ickmeans)
centers = plot_knee(clmax,sseclusters,PDIR)


print(f'Centroids ideais: {centers}')
# Definindo metodos e núcleos #
print('12 - Definindo metodos e núcleos')
method = 'lloyd'
centroids = centers
#centroids = 12
print(f'Metodo: {method}')
print(f'Núcleos: {centroids}')
print('13 - Rodando Kmeans')
    

# Opcao 1 do K-Means sem normalizacao #
u1, v1, hgt1, cluster_fraction1, lat, lon, mclusters1, temp = camins_update(centroids,3,100,
                                                                        ds_u_sel,
                                                                        ds_v_sel,
                                                                        ds_hgt_sel,
                                                                        method, 
                                                                        scaler_type=None, 
                                                                        joint_scaling='True')


# Opcao 2 do K-Means com normalizacao #
#u2, v2, hgt2, cluster_fraction2, lat, lon, mclusters2, temp2 = camins_update(centroids,3,100,
                                                                        #ds_u2_sel,
                                                                        #ds_v2_sel,
                                                                        #ds_hgt2_sel,
                                                                        #method, 
                                                                        #scaler_type='standard', 
                                                                        #joint_scaling='True')



# Plotando #
print('14 - Plotando clusters')



directory = plota_clusters_extreme(u1, v1, hgt1, lat, lon, 
                                                    cluster_fraction1,lat_boia, lon_boia,PDIR)


#dir2 = plota_clusters_extreme2(u2, v2, hgt2, lat, lon,
                                                    #cluster_fraction2,PDIR)

# Pegando dias de cada cluster #
print('15 - Pegando dias de cada cluster')
wtdays_dict = pega_dias(mclusters1,temp,cluster_fraction1)
#wtdays_dict2 = pega_dias(mclusters2,temp2,cluster_fraction2)

monthly_counts(wtdays_dict, PDIR, centroids)

#  Salvando dados #
print('16 - Salvando dados')
save_array(u1,'u_kmeans',PDIR)
save_array(v1,'v_kmeans',PDIR)
save_array(hgt1,'hgt_kmeans',PDIR)
save_array(cluster_fraction1,'cluster_fraction',PDIR)
save_dict(wtdays_dict,'wtdays_dict',PDIR)
save_kmeans(mclusters1,'mclusters',PDIR)
save_array(lat,'lat',PDIR)
save_array(lon,'lon',PDIR)
save_array(temp,'tempo',PDIR)

# Plotando clusters com dias de ondas extremas #
print('17 - Plotando clusters com dias de ondas extremas')


plota_clusters_extreme_wave_rose(ds_hs,ds_tp,ds_dir,wtdays_dict,u1, v1, hgt1, 
                                lat, lon, cluster_fraction1,lon_boia, lat_boia, boia, PDIR)


#wtdays_dict = pega_dias(mclusters1,temp,cluster_fraction1)
wtdays_dict24 = pega_dias24(mclusters1,temp,cluster_fraction1)
wtdays_dict48 = pega_dias48(mclusters1,temp,cluster_fraction1)
save_dict(wtdays_dict24,'wtdays_dict24',PDIR)
save_dict(wtdays_dict48,'wtdays_dict48',PDIR)



u_2days, v_2days, hgt_2days = before_days(wtdays_dict48, ds_u, ds_v, ds_hgt)
u_1day, v_1day, hgt_1day = before_days(wtdays_dict24, ds_u, ds_v, ds_hgt)



plota_clusters_extreme_waves_before(ds_hs,ds_tp,ds_dir,wtdays_dict,u1, v1, hgt1, lat, lon, u_1day, v_1day, hgt_1day, 
                                    u_2days, v_2days, hgt_2days, cluster_fraction1,lon_boia, lat_boia, boia,PDIR)