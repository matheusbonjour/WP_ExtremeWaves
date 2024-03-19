
import numpy as np 
import os
import pandas as pd
import pathlib
from scipy import signal, stats
import scipy.io 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import string
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import xarray as xr 

# Local module imports, assuming these are custom modules and not standard libraries
from boia_domain_disserta import CreateMapBoia
from GetDays import pega_dias, pega_dias24, pega_dias48
from KneeLocWP import prekneed1, teste_kmeans, plot_knee

# Warning suppression and possible custom library imports (commented out because they seem to be custom or unavailable in standard libraries)
warnings.filterwarnings("ignore")


def le_dados_onda():
    # Le arquivo com dados de Hs, Tp e Dp

    #onda_file = '../../wave_wtdomain_1993_2019.nc'
    #onda_file = '../../waverys9318_maior.nc'
    onda_file = '../data/final_waverys.nc'

    #onda_file = '../data/merged_final_waverys9323.nc'
    
    ds_onda = xr.open_dataset(onda_file,decode_times=True)
    
    
    return ds_onda



def le_dados_era():
    #leitura dos dados
    #u_file2 = '../../era5_u1000_9319.nc'
    #v_file2 = '../../era5_v1000_9319.nc'
    #hgt_file2 =  '../../era5_hgt1000_9319.nc'

    #u_file2 = '../../u_maior_9319_daymean.nc'
    #v_file2 = '../../v_maior_9319_daymean.nc'
    #hgt_file2 =  '../../hgt_maior_9319_daymean.nc'

    u_file2 = '../data/final_ERA5_u.nc'
    v_file2 = '../data/final_ERA5_v.nc'
    hgt_file2 =  '../data/final_ERA5_hgt.nc'

    #u_file2 = '../../merged_final_u_9323.nc'
    #v_file2 = '../../merged_final_v_9323.nc'
    #hgt_file2 =  '../../merged_final_hgt_9323.nc'

    ds_u = xr.open_dataset(u_file2,decode_times=True)
    ds_v = xr.open_dataset(v_file2,decode_times=True)
    ds_hgt = xr.open_dataset(hgt_file2,decode_times=True)

    return ds_u, ds_v, ds_hgt

def sel_var_onda(ds_onda,str_variable):
    strvar = str(str_variable)
    onda_var = ds_onda[{strvar}]
    ds_strvar = onda_var
    return ds_strvar

def sel_time_onda(ds_onda,start,end):
    ds_time_onda = ds_onda.sel(time=slice(start,end))

    #ds_time_onda = ds_time_onda.resample(time='D').mean()

    return ds_time_onda

def sel_time_era(ds_u,ds_v,ds_hgt,start,end):
    ds_u = ds_u['u'].sel(time=slice(start,end))
    ds_v = ds_v['v'].sel(time=slice(start,end))
    ds_hgt = ds_hgt['z'].sel(time=slice(start,end))

    #ds_u = ds_u.resample(time='D').mean()
    #ds_v = ds_v.resample(time='D').mean()
    #ds_hgt = ds_hgt.resample(time='D').mean()


    return ds_u, ds_v, ds_hgt

def sel_time_era_extreme(ds_u,ds_v,ds_hgt,extreme_days):
    ds_u = ds_u.sel(time=extreme_days).squeeze()
    ds_v = ds_v.sel(time=extreme_days).squeeze()
    ds_hgt = ds_hgt.sel(time=extreme_days).squeeze()

    return ds_u, ds_v, ds_hgt

def convert_vel(dsk_dir):
    dp2=np.array(dsk_dir.copy())
    ol=np.where((dsk_dir>=0) & (dsk_dir<180))
    dp2[ol]=dp2[ol] + 180
    ol=np.where((dsk_dir>=180) &  (dsk_dir<360))
    dp2[ol]=dp2[ol] - 180
    ol=np.where(np.isnan(dp2))
    dp2[ol]=0
    dsk_uw=np.zeros(dsk_dir.shape)
    dsk_vw=np.zeros(dsk_dir.shape)
    for jk in range(dsk_uw.shape[0]):
        for kj in range(dsk_uw.shape[1]):
            dsk_uw[jk,kj], dsk_vw[jk,kj] = vel_conv(1,dp2[jk,kj])    


    return dsk_uw, dsk_vw 

def vel_conv(vel,dir):
    """Converts a velocity vector to u,v components.
    
    Parameters
    ----------
    vel : float
        Velocity magnitude.
    dir : float
        Velocity direction in degrees.
    
    Returns
    -------
    u : float
        Velocity in u-direction.
    v : float
        Velocity in v-direction.
    """
    
    # Calculate u and v components for each quadrant
    if dir <= 90:
        u = vel*np.sin(np.radians(dir))
        v = vel*np.cos(np.radians(dir))
    if dir > 90 and dir <=180:
        dir=dir-90
        u = vel*np.cos(np.radians(dir))
        v = -vel*np.sin(np.radians(dir))
    if dir > 180 and dir <=270:
        dir=dir-180
        u = -vel*np.sin(np.radians(dir))
        v = -vel*np.cos(np.radians(dir))
    if dir > 270 and dir <=360:
        dir=dir-270
        u = -vel*np.cos(np.radians(dir))
        v = vel*np.sin(np.radians(dir))
    return(u,v) 




def sel_point_lat_lon(ds_onda, lat, lon):

    ds_onda_point = ds_onda.sel(latitude=lat,longitude=lon,method='nearest').squeeze()
    return ds_onda_point



def sel_wave_days(wtdays, ds_hs, ds_tp, ds_dir):

    ds_hs_sel = ds_hs['VHM0'].sel(time=wtdays).squeeze()
    ds_tp_sel = ds_tp['VTPK'].sel(time=wtdays).squeeze()
    ds_dir_sel = ds_dir['VMDR'].sel(time=wtdays).squeeze()

    return ds_hs_sel, ds_tp_sel, ds_dir_sel 



def get_point_lat_lon(buoy,PDIR):
    # Diretorio dos dados
    #dataset_path = "../../ETOPO1_Bed_g_gmt4.grd"


    region = 'atlsul'
    #buoy = 'vitoria'
    

    mapa = CreateMapBoia(region, buoy, PDIR)

    return mapa.get_lat_lon()




def before_days(dicionario, u, v, hgt):
    list_u = []
    list_v = []
    list_hgt = []

    for key in dicionario:

        datas = pd.to_datetime(dicionario[key])


        u_selecionado = u.sel(time=datas)
        v_selecionado = v.sel(time=datas)
        hgt_selecionado = hgt.sel(time=datas)

        u_selecionado = u_selecionado.mean('time')
        v_selecionado = v_selecionado.mean('time')
        hgt_selecionado = hgt_selecionado.mean('time')


        u_selecionado = u_selecionado.assign_coords(time=key)
        v_selecionado = v_selecionado.assign_coords(time=key)
        hgt_selecionado = hgt_selecionado.assign_coords(time=key)

        list_u.append(u_selecionado)
        list_v.append(v_selecionado)
        list_hgt.append(hgt_selecionado)

    u_combined = xr.concat(list_u, dim='time')
    v_combined = xr.concat(list_v, dim='time')
    hgt_combined = xr.concat(list_hgt, dim='time')

    return u_combined, v_combined, hgt_combined


def convert_array(ds_u,ds_v,ds_hgt):
    ds_u_arr = ds_u.to_array().squeeze()
    ds_v_arr = ds_v.to_array().squeeze()
    ds_hgt_arr = ds_hgt.to_array().squeeze() 

    return ds_u_arr, ds_v_arr, ds_hgt_arr 

