import json
import string
from boia2_domain import CreateMapBoia
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import pandas as pd
import numpy as np 
import math as m
import scipy.io 
import sys, glob
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import os
import cartopy, cartopy.crs as ccrs 
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from kneed import KneeLocator
from scipy import signal
import warnings
#import cmocean 
warnings.filterwarnings("ignore")
#from eofs.xarray import Eof
#from eofs.multivariate.standard import MultivariateEof
import pathlib
from datetime import datetime, timedelta
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import xarray as xr 
import calendar
import matplotlib.dates as mdates
from KneeLocWP import prekneed1, teste_kmeans, plot_knee
from GetDays import pega_dias, pega_dias24, pega_dias48
import matplotlib.image as mpimg
import joblib
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm

def le_dados_onda():
    # Le arquivo com dados de onda
    #onda_file = '../../wave_wtdomain_1993_2019.nc'
    #onda_file = '../../waverys9318_maior.nc'
    onda_file = '../../final_waverys.nc'
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

    u_file2 = '../../final_ERA5_u.nc'
    v_file2 = '../../final_ERA5_v.nc'
    hgt_file2 =  '../../final_ERA5_hgt.nc'
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
    dataset_path = "../../ETOPO1_Bed_g_gmt4.grd"


    region = 'atlsul'
    #buoy = 'vitoria'
    

    mapa = CreateMapBoia(dataset_path, region, buoy, PDIR)

    return mapa.get_lat_lon()


def camins(ncenters,ninit,maxiter,u,v,hgt,algo):
    """ 
    Recebe o numero de clustes, numero de inicializações, numero maximo de iterações
    e os datasets de u, v e hgt. 
    
    Returna os x centros em dataset dos campos de u, v e hgt 
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
    dsk_means = np.concatenate((dsk_u,dsk_v,dsk_hgt), axis=1)
    #scaler = StandardScaler()
    #scaled_dskmeans = scaler.fit_transform(dsk_means)
    scaled_dskmeans = dsk_means

    mk = KMeans(n_clusters=ncenters,algorithm=algo).fit(scaled_dskmeans)
    
    def get_cluster_fraction(m, label):
        kf = (m.labels_==label).sum()/(m.labels_.size*1.0)
        return kf*100 
    
    slcenter = len(lat)*len(lon) 
    centers_u = mk.cluster_centers_[:,0:slcenter]
    centers_v = mk.cluster_centers_[:,slcenter:slcenter*2]
    centers_hgt = mk.cluster_centers_[:,slcenter*2:slcenter*3]
    
    u_wt = np.zeros((ncenters,ny,nx)) 
    v_wt = np.zeros((ncenters,ny,nx)) 
    hgt_wt = np.zeros((ncenters,ny,nx)) 
    cf = np.zeros((ncenters)) 
    for i in range(mk.n_clusters):    
        u_wt[i] = centers_u[i,:].reshape(ny,nx, order='F')
        v_wt[i] = centers_v[i,:].reshape(ny,nx, order='F')
        hgt_wt[i] = centers_hgt[i,:].reshape(ny,nx, order='F')
        cf[i] = get_cluster_fraction(mk, i)
    
    #return u_wt, v_wt, hgt_wt 
    return u_wt, v_wt, hgt_wt, cf, lat, lon, mk, tempo



def camins_update(ncenters, ninit, maxiter, u, v, hgt, algo, scaler_type='none', joint_scaling=True):
    """ 
    Recebe o numero de clustes, numero de inicializações, numero maximo de iterações
    e os datasets de u, v e hgt. 
    
    Returna os x centros em dataset dos campos de u, v e hgt 
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


def plota_clusters_extreme(u_kmeans, v_kmeans, hgt_kmeans, lat, lon, cluster_fraction,lat_boia, lon_boia,PDIR):
    """
    Plota os clusters de u, v e hgt em um mesmo plot
    """

    lat = lat
    lon = lon
    latmin = lat.min().values
    latmax = lat.max().values
    lonmin = lon.min().values
    lonmax = lon.max().values
    ncenters = len(cluster_fraction)
    nline = int(np.ceil(ncenters / 2.0))
    regimes = []
    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
    
    tags = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 
            'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 'u)', 'v)', 'w)', 'x)', 'y)', 'z)', 
            'aa)', 'bb)', 'cc)', 'dd)', 'ee)', 'ff)', 'gg)', 'hh)', 'ii)', 'jj)', 'kk)', 'll)',
            'mm)', 'nn)', 'oo)', 'pp)', 'qq)', 'rr)', 'ss)', 'tt)', 'uu)', 'vv)', 'ww)', 'xx)', 'yy)', 'zz)'
            'aaa)', 'bbb)', 'ccc)', 'ddd)', 'eee)', 'fff)', 'ggg)', 'hhh)', 'iii)', 'jjj)', 'kkk)', 'lll)']

    extent = [lonmin, lonmax, latmin, latmax] 
    img_extent = [extent[0], extent[2], extent[1], extent[3]]
    #data_min = hgt_kmeans.min()/2
    #data_max = hgt_kmeans.max()
    #interval = 10
    #levels = np.arange(data_min,data_max,interval)

    max_abs_value = max(-np.min(hgt_kmeans), np.max(hgt_kmeans)) - 250
    
    # Definindo os níveis de contorno para serem centrados em zero
    levels = np.linspace(-max_abs_value, max_abs_value, num=21) 

    sp = 12
    fig = plt.figure(figsize=((9.6), (2*ncenters)+2))
    gs = gridspec.GridSpec(nline, 2, figure=fig,wspace=-0.4, hspace=0.15)

    for i in range(ncenters):

        if i == ncenters - 1 and ncenters % 2 != 0:
            ax = fig.add_subplot(gs[i//2, :], projection=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(gs[i//2, i%2], projection=ccrs.PlateCarree())

        #ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
        cs = ax.contourf(lon, lat, hgt_kmeans[i], levels=levels, extend='both',cmap='RdBu_r')

        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 10), ylocs=np.arange(-90, 90, 10), draw_labels=True)
        gl.xlabel_style = {'fontsize': 13, 'fontweight': 'bold'}
        gl.ylabel_style = {'fontsize': 13, 'fontweight': 'bold'}
        gl.top_labels = False
        gl.right_labels = False 

        if i % 2 == 1:
            gl.left_labels = False
            gl.right_labels = False 
            gl.top_labels = False

        if ncenters > 4:
            if i == 0 or i == 1 or i == 2 or i == 3:
                gl.bottom_labels = False
                gl.top_labels = False
        
        if ncenters <= 4:
            if i == 0 or i == 1:
                gl.bottom_labels = False
                gl.top_labels = False

        ww = ax.quiver(lon[::sp], lat[::sp], u_kmeans[i,::sp,::sp], v_kmeans[i,::sp,::sp],headwidth=4, headlength=4, headaxislength=4, scale=100)
        ax.quiverkey(ww, 0.87, 0.92, 10,r'$10 m/s $', labelpos='N', coordinates='figure',fontproperties={'size': 14})
        title = '{}, {:4.1f}%'.format(regimes[i], cluster_fraction[i])



        ax.plot(lon_boia, lat_boia, marker='o', markersize=7, color='lime', alpha=0.8, markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree())

        ax.set_title(title, fontweight='bold', fontsize=16)
        plt.text(-0.1, 1, tags[i], 
             transform=ax.transAxes, 
             va='bottom', 
             fontsize=plt.rcParams['font.size']*2, 
             fontweight='bold')   


    cbar_ax = fig.add_axes([0.85, 0.05, 0.055, 0.85])
    cb = fig.colorbar(cs, cax=cbar_ax, shrink=0.8, aspect=20)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('Geopotential Height (m)',labelpad=-3,fontsize=16,fontweight='bold') 
    plt.subplots_adjust(left=-0.10,right=0.99,bottom=0.05,top=0.96)
    #plt.tight_layout()

    PDIR2=PDIR+'figures/'
    pasta = pathlib.Path(PDIR2)
    pasta.mkdir(parents=True, exist_ok=True)
    fname2 = str(ncenters)+"patterns2.png"
    
    plt.savefig(PDIR2+fname2,dpi=300)  
    plt.close()


def plot_contourf(ax, lon, lat, data, j, levels, cmap, ncenters, lon_boia, lat_boia):
    csa = ax.contourf(lon, lat, data[j], levels=levels, extend='both',cmap=cmap)
    ax.plot(lon_boia, lat_boia, marker='o', markersize=10, color='lime', alpha=0.8, markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=100)

    ax.coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.xlabel_style = {'fontsize': 14, 'fontweight': 'bold'}
    gl.ylabel_style = {'fontsize': 14, 'fontweight': 'bold'}
    gl.top_labels = False
    gl.right_labels = False 

    if j != ncenters-1:
        gl.bottom_labels = False

    return csa 

def plot_contourf_teste(ax, lon, lat, data, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia):
    csa = ax.contourf(lon, lat, data[j], levels=levels, extend='both',cmap=cmap)
    ax.plot(lon_boia, lat_boia, marker='o', markersize=11, color='lime', alpha=0.8, markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=100)

    ax.coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.xlabel_style = {'fontsize': 14, 'fontweight': 'bold'}
    gl.ylabel_style = {'fontsize': 14, 'fontweight': 'bold'}
    gl.top_labels = False
    gl.right_labels = False 
    if i % cols != 0:
        gl.left_labels = False
    

    if j != ncenters-1:
        gl.bottom_labels = False

    return csa 

def plot_contourf_hs(ax, lon, lat, dsk_hs, levelshs, cmap1, norm, j, ncenters, lon_boia, lat_boia):
    csw = ax.contourf(lon, lat, dsk_hs[:,:], levels=levelshs, extend='both',cmap=cmap1, norm=norm)
    ax.plot(lon_boia, lat_boia, marker='o', markersize=11, color='lime', alpha=0.8, markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=100)
            
    ax.coastlines(resolution='50m', color='black', linewidth=0.8,zorder=2)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5,zorder=5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.xlabel_style = {'fontsize': 14, 'fontweight': 'bold'}
    gl.ylabel_style = {'fontsize': 14, 'fontweight': 'bold'}
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    if j != ncenters-1:
        gl.bottom_labels = False
    
    return csw 

def plot_quiver_hs(ax, lonwave, latwave, dsk_uw, dsk_vw, sp):
    ww = ax.quiver(lonwave[::sp], latwave[::sp], dsk_uw[::sp, ::sp], dsk_vw[::sp, ::sp],headwidth=3, headlength=3, headaxislength=3, scale=25,zorder=2)
    ax.add_feature(cfeature.LAND,zorder=3)

def insert_title_hs(ax, tags, i, j, regimes):
    title = 'Wave mean: {} days'.format(regimes[j])
    ax.set_title(title, fontsize=plt.rcParams['font.size']*1.7, fontweight='bold')
    plt.text(-0.05, 1, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')


def plot_quiver(ax, lon, lat, u, v, j, sp):
    ww = ax.quiver(lon[::sp], lat[::sp], u[j,::sp,::sp], v[j,::sp,::sp],headwidth=4, headlength=4, headaxislength=4, scale=100)
    return ww

def insert_title(ax, tags, i, j, cluster_fraction2, regimes):
    title = '{}, {:4.1f}%'.format(regimes[j], cluster_fraction2[j])
    ax.set_title(title, fontsize=plt.rcParams['font.size']*3, fontweight='bold', color='red')
    plt.text(-0.05, 1, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')
    
def insert_title24(ax, tags, i, j, cluster_fraction2, regimes):
    title = '24h before {}'.format(regimes[j])
    ax.set_title(title, fontsize=plt.rcParams['font.size']*2, fontweight='bold')
    plt.text(-0.05, 1, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')

def insert_title48(ax, tags, i, j, cluster_fraction2, regimes):
    title = '48h before {}'.format(regimes[j])
    ax.set_title(title, fontsize=plt.rcParams['font.size']*2, fontweight='bold')
    plt.text(-0.05, 1, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')


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

def cmap_wave():
     # Lista de cores em hexadecimal
    linear_gradient = [
        (144, 240, 242),
        (71, 221, 230),
        (49, 145, 176),
        (53, 102, 175),
        (56, 72, 156),
        (126, 49, 59),
        (192, 123, 111),
        (192, 162, 157),
    ]

    colors_rgb_normalized = [(r/255, g/255, b/255) for r, g, b in linear_gradient]


    cmapwave = LinearSegmentedColormap.from_list("my_colormap", colors_rgb_normalized)

    return cmapwave 




def plota_clusters_extreme_waves2(ds_hs,ds_tp,ds_dir,wtdays_dict1,u_kmeans1, v_kmeans1, hgt_kmeans1, 
                                  lat, lon, cluster_fraction2,lon_boia, lat_boia, boia, PDIR):
    # Função para converter RGB para hexadecimal

    # Criando a paleta de cores personalizada
    #windy_wave_cmap = LinearSegmentedColormap.from_list("paleta_rgb", colors_hex)
    
    cmap1 = cmap_wave()
    ###################################################################
    lat = lat
    lon = lon
    latwave = ds_hs.latitude.values
    lonwave = ds_hs.longitude.values

    if boia == 'vitoria':
        factor = 1.3
    
    if boia == 'santos':
        factor = 1.3
    
    if boia == 'riogrande':
        factor = 1.3

    if boia == 'campos':
        factor = 1.3 

    
    data_minhs = np.nanmin(ds_hs.VHM0.mean('time'))
    data_maxhs = np.nanmax(ds_hs.VHM0.mean('time'))+factor
    intervalhs = 0.5
    data_minhs = round(data_minhs,1)
    data_maxhs = round(data_maxhs,1)
    levelshs = np.arange(data_minhs,data_maxhs,intervalhs)
    #boundaries = np.arange(data_minhs, data_maxhs, 0.5)
    boundaries = np.arange(data_minhs, data_maxhs + 0.5, 0.5)
    norm = BoundaryNorm(boundaries, cmap1.N, clip=True)
    latmin = lat.min().values
    latmax = lat.max().values
    lonmin = lon.min().values
    lonmax = lon.max().values
    ncenters = len(cluster_fraction2)
    nline = int(ncenters/2)

    regimes = []
    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
        nome_arquivo = "WP+WAVE1.png"


    tags = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 
            'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 'u)', 'v)', 'w)', 'x)', 'y)', 'z)', 
            'aa)', 'bb)', 'cc)', 'dd)', 'ee)', 'ff)', 'gg)', 'hh)', 'ii)', 'jj)', 'kk)', 'll)',
            'mm)', 'nn)', 'oo)', 'pp)', 'qq)', 'rr)', 'ss)', 'tt)', 'uu)', 'vv)', 'ww)', 'xx)', 'yy)', 'zz)'
            'aaa)', 'bbb)', 'ccc)', 'ddd)', 'eee)', 'fff)', 'ggg)', 'hhh)', 'iii)', 'jjj)', 'kkk)', 'lll)']




    data_min = hgt_kmeans1.min()/2
    data_max = hgt_kmeans1.max()
    interval = 10
    sp = 12
    max_abs_value = max(-np.min(hgt_kmeans1), np.max(hgt_kmeans1)) - 250
    # Definindo os níveis de contorno para serem centrados em zero
    levels = np.linspace(-max_abs_value, max_abs_value, num=21)  # altere o num para o número de níveis desejado

 
    fig_nums = m.ceil(ncenters / 3.0) # arredonda para cima
    figs = []
    name_fig = []
    rows = ncenters
    cols = 2

    fig = plt.figure(figsize=((12), (ncenters*7)-2))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=-0.09, hspace=0.1)

    
    
    #rows = 2 if ncenters == 4 else 3

    for i in range(rows*cols):
        ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())

        if i % 2 == 0:
            j = i // 2
            cmap='RdBu_r'
            csa = plot_contourf(ax, lon, lat, hgt_kmeans1, j, levels, cmap, ncenters, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_kmeans1, v_kmeans1, j, sp)
            insert_title(ax, tags, i, j, cluster_fraction2, regimes)
        

        # Coluna 1 - Waves 
        else:
            j = (i - 1) // 2
            
            wtstr = regimes[j]
            wt_days = wtdays_dict1[wtstr]
            ds_hs_sel, ds_tp_sel, ds_dir_sel = sel_wave_days(wt_days, ds_hs, ds_tp, ds_dir)
            if ds_hs_sel.shape[0] > 1:
                dsk_hs = ds_hs_sel.mean('time').values
                dsk_dir = ds_dir_sel.mean('time').values
            elif ds_hs_sel.shape[0] == 1:
                dsk_hs = ds_hs_sel.values
                dsk_dir = ds_dir_sel.values

            dsk_uw, dsk_vw = convert_vel(dsk_dir)
            csw = plot_contourf_hs(ax, lonwave, latwave, dsk_hs, levelshs, cmap1, norm, j, ncenters, lon_boia, lat_boia)
            plot_quiver_hs(ax, lonwave, latwave, dsk_uw, dsk_vw, sp)
            insert_title_hs(ax, tags, i, j, regimes)


    cbar_ax_atm= fig.add_axes([0.05, 0.025, 0.420, 0.02])
    cbar_ax_wav= fig.add_axes([0.535, 0.025, 0.420, 0.02])

    cba = fig.colorbar(csa, cax=cbar_ax_atm, shrink=0.8, aspect=20, orientation='horizontal')
    cba.ax.tick_params(labelsize=12)
    cba.set_label('Geopotencial Height(m)',labelpad=0,fontsize=18,fontweight='bold') 

    cbw = fig.colorbar(csw, cax=cbar_ax_wav, shrink=0.8, aspect=20, orientation='horizontal') 
    cbw.ax.tick_params(labelsize=12)
    cbw.set_label('Significant Wave Height (m)',labelpad=0,fontsize=20,fontweight='bold')

    cba.locator = ticker.MaxNLocator(nbins=4)
    cba.update_ticks()
    # method1
    ticks = boundaries[0:-1]
    cbw.set_ticks(ticks)


    # method2
    #cbw.locator = MultipleLocator(base=0.5)
    #cbw.update_ticks()
    
    # method3
    #cbw.set_ticks(boundaries + 0.25)
    #cbw.set_ticklabels(boundaries)

    plt.subplots_adjust(left=0.02,right=1,bottom=0.063,top=0.98)
  
    PDIR2 = PDIR +'/figures/'
    pasta = pathlib.Path(PDIR2)
    ind = 1 


    pasta.mkdir(parents=True, exist_ok=True)
    fname2 = f'WP+WAVE+{boia}.png'
    plt.savefig(PDIR2+fname2,dpi=300)  
    plt.close()


def plota_clusters_extreme_waves2_teste(ds_hs,ds_tp,ds_dir,wtdays_dict1,u_kmeans1, v_kmeans1, hgt_kmeans1, lat, lon, u_1day, v_1day, hgt_1day, 
                                        u_2days, v_2days, hgt_2days, cluster_fraction2,lon_boia,lat_boia, boia,PDIR):
    # Função para converter RGB para hexadecimal

    # Criando a paleta de cores personalizada
    #windy_wave_cmap = LinearSegmentedColormap.from_list("paleta_rgb", colors_hex)
    if boia == 'vitoria':
        factor = 1.3
    
    if boia == 'santos':
        factor = 1.3
    
    if boia == 'riogrande':
        factor = 1.3
    if boia == 'campos':
        factor = 1.3 

    cmap1 = cmap_wave()
    ###################################################################
    lat = lat
    lon = lon
    latwave = ds_hs.latitude.values
    lonwave = ds_hs.longitude.values
    data_minhs = np.nanmin(ds_hs.VHM0.mean('time'))
    data_maxhs = np.nanmax(ds_hs.VHM0.mean('time'))+factor
    intervalhs = 0.5
    data_minhs = round(data_minhs,1)
    data_maxhs = round(data_maxhs,1)
    levelshs = np.arange(data_minhs,data_maxhs,intervalhs)
    boundaries = np.arange(data_minhs, data_maxhs + 0.5, 0.5)
    norm = BoundaryNorm(boundaries, cmap1.N, clip=True)
    latmin = lat.min().values
    latmax = lat.max().values
    lonmin = lon.min().values
    lonmax = lon.max().values
    ncenters = len(cluster_fraction2)
    nline = int(ncenters/2)

    regimes = []
    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
        nome_arquivo = "WP+WAVE1.png"

    

    # Cria uma lista para armazenar as tags
    tags = []

    # Primeiro adiciona as letras únicas
    for letter in string.ascii_lowercase:
        tags.append(letter + ')')

    # Então adiciona as combinações de duas letras
    for first_letter in string.ascii_lowercase:
        for second_letter in string.ascii_lowercase:
            tags.append(first_letter + second_letter + ')')
    #tags = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)', 't)', 'u)', 'v)', 'w)', 'x)', 'y)', 'z)']




    data_min = hgt_kmeans1.min()/2
    data_max = hgt_kmeans1.max()
    interval = 10
    sp = 12
    max_abs_value = max(-np.min(hgt_kmeans1), np.max(hgt_kmeans1)) - 250
    # Definindo os níveis de contorno para serem centrados em zero
    levels = np.linspace(-max_abs_value, max_abs_value, num=21)  # altere o num para o número de níveis desejado

 
    fig_nums = m.ceil(ncenters / 3.0) # arredonda para cima
    figs = []
    name_fig = []
    rows = ncenters
    cols = 4

    fig = plt.figure(figsize=((6*cols)-1, (ncenters*7)-2))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=-0.09, hspace=0.1)

    
    
    #rows = 2 if ncenters == 4 else 3

    for i in range(rows*cols):
        ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())

        # Coluna 0 - HGT 2 dias antes
        if i % cols == 0:
            j = i // cols
            cmap='RdBu_r'
            csa = plot_contourf_teste(ax, lon, lat, hgt_2days, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_2days, v_2days, j, sp)
            insert_title48(ax, tags, i, j, cluster_fraction2, regimes)
        
        # 
        elif i % cols == 1:
            j = i // cols
            cmap='RdBu_r'
            csa = plot_contourf_teste(ax, lon, lat, hgt_1day, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_1day, v_1day, j, sp)
            insert_title24(ax, tags, i, j, cluster_fraction2, regimes)

        elif i % cols == 2:
            j = i // cols
            cmap='RdBu_r'
            csa = plot_contourf_teste(ax, lon, lat, hgt_kmeans1, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_kmeans1, v_kmeans1, j, sp)
            insert_title(ax, tags, i, j, cluster_fraction2, regimes)
        
        elif i % cols == 3:
            j = (i - 1) // cols
            
            wtstr = regimes[j]
            wt_days = wtdays_dict1[wtstr]
            ds_hs_sel, ds_tp_sel, ds_dir_sel = sel_wave_days(wt_days, ds_hs, ds_tp, ds_dir)
            if ds_hs_sel.shape[0] > 1:
                dsk_hs = ds_hs_sel.mean('time').values
                dsk_dir = ds_dir_sel.mean('time').values
            elif ds_hs_sel.shape[0] == 1:
                dsk_hs = ds_hs_sel.values
                dsk_dir = ds_dir_sel.values

            dsk_uw, dsk_vw = convert_vel(dsk_dir)
            csw = plot_contourf_hs(ax, lonwave, latwave, dsk_hs, levelshs, cmap1, norm, j, ncenters, lon_boia, lat_boia)
            plot_quiver_hs(ax, lonwave, latwave, dsk_uw, dsk_vw, sp)
            insert_title_hs(ax, tags, i, j, regimes)


    cbar_ax_atm= fig.add_axes([0.05, 0.025, 0.420, 0.02])
    cbar_ax_wav= fig.add_axes([0.535, 0.025, 0.420, 0.02])

    cba = fig.colorbar(csa, cax=cbar_ax_atm, shrink=0.8, aspect=20, orientation='horizontal')
    cba.ax.tick_params(labelsize=12)
    cba.set_label('Geopotencial Height(m)',labelpad=0,fontsize=18,fontweight='bold') 

    cbw = fig.colorbar(csw, cax=cbar_ax_wav, shrink=0.8, aspect=20, orientation='horizontal') 
    cbw.ax.tick_params(labelsize=12)
    cbw.set_label('Significant Wave Height (m)',labelpad=0,fontsize=20,fontweight='bold')

    cba.locator = ticker.MaxNLocator(nbins=4)
    cba.update_ticks()
    ticks = boundaries[:-1]
    #cbw.locator = MultipleLocator(base=0.5)
    #cbw.update_ticks()
    cbw.set_ticks(ticks)

    plt.subplots_adjust(left=0.02,right=1,bottom=0.063,top=0.98)

    PDIR2 = PDIR +'/figures/'
    pasta = pathlib.Path(PDIR2)
    ind = 1 


    pasta.mkdir(parents=True, exist_ok=True)
    fname2 = f'WP+WAVE_past_+{boia}.png'
    plt.savefig(PDIR2+fname2,dpi=300)  
    plt.close()





def plot_hs_serie(ds_wave_point,perc,percentil, PDIR):

    #ig = plt.figure(figsize=(14,8))
    
    #ax = fig.add_subplot(111)
    fig = plt.figure(figsize=((10, 6)))
    gs = gridspec.GridSpec(1,1, figure=fig,wspace=0, hspace=0)
    ax = fig.add_subplot(gs[0, :])
    
    ax.plot(ds_wave_point['time'],ds_wave_point['VHM0'].values, color='#0001C2', linewidth=1)
    ax.axhline(y=perc, color='#5A5857', linestyle='-', linewidth=6)
    # formanto perc para apresentar apenas 2 casas decimais
    perc = '{:.2f}'.format(perc)
    ax.legend(['Daily Mean',f'{percentil}-percentile = {perc}'],loc='upper right',fontsize=12)

    ax.set_ylabel('Significant Wave Height (m)', fontsize=14, weight='bold')
    ax.set_xlabel('Time', fontsize=14, weight='bold')
    # Aumentando fontsize dos labels e colocando negrito
    ax.set_xticklabels(pd.date_range(ds_wave_point['time'].values[0],ds_wave_point['time'].values[-1],freq='2Y'), rotation=45, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    years_fmt = mdates.DateFormatter('%Y')
    years = mdates.YearLocator(base=2) 
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    fig.autofmt_xdate()
    # Obter limites do eixo y
    y_min, y_max = ax.get_ylim()
    yticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)
    ax.set_yticks(yticks)
    #ax.set_yticklabels(yticks, fontdict={'fontsize': 13, 'fontweight': 'bold'})
    ax.set_yticklabels(yticks.astype(int), fontdict={'fontsize': 13, 'fontweight': 'bold'})




    # Aumentando fontsize dos ticks e colocando negrito
    ax.set_xlim(ds_wave_point['time'].values[0],ds_wave_point['time'].values[-1])
    #ax.set_facecolor('#EBE7C0')
    ax.grid(True, color='#696969', linestyle='-', linewidth=0.4)
    
    # Subplot adjust
    plt.subplots_adjust(left=0.05,right=0.98,bottom=0.15,top=0.95)


    PDIR2 = PDIR +'/figures/'
    pasta = pathlib.Path(PDIR2)
    pasta.mkdir(parents=True, exist_ok=True)

    plt.savefig(PDIR2+'hs_serie.png',dpi=300)
    plt.close()



def convert_array(ds_u,ds_v,ds_hgt):
    ds_u_arr = ds_u.to_array().squeeze()
    ds_v_arr = ds_v.to_array().squeeze()
    ds_hgt_arr = ds_hgt.to_array().squeeze() 

    return ds_u_arr, ds_v_arr, ds_hgt_arr 

def save_array(variavel, nome, PDIR):
    PDIR2 = PDIR+'data/'
    if not os.path.exists(PDIR2):
        os.makedirs(PDIR2)

    np.save(PDIR2+nome,variavel)

def save_kmeans(mkobject, nome, PDIR):
    PDIR2 = PDIR+'data/'
    if not os.path.exists(PDIR2):
        os.makedirs(PDIR2)

    joblib.dump(mkobject, PDIR2+nome+'.pkl')


def save_dict(dict_data, nome, PDIR):
    PDIR2 = PDIR+'data/'
    if not os.path.exists(PDIR2):
        os.makedirs(PDIR2)

    # Converta arrays numpy e datas para listas e strings, respectivamente
    converted_data = {key: [np.datetime_as_string(i, unit='ns') for i in value] for key, value in dict_data.items()}

    with open(os.path.join(PDIR2, nome + '.json'), 'w') as file:
        json.dump(converted_data, file)

def load_dict(nome, PDIR):
    with open(os.path.join(PDIR, nome + '.json'), 'r') as file:
        loaded_data = json.load(file)
        
    # Converta strings para datas e listas para arrays numpy
    converted_data = {key: np.array([np.datetime64(i) for i in value]) for key, value in loaded_data.items()}
    
    return converted_data

def monthly_counts(wtdays_dict, PDIR, ncenters):
    wtdays_dict_monthly = {}
    for wp, dates in wtdays_dict.items():
        dates = pd.to_datetime(dates)
        wtdays_dict_monthly[wp] = dates.month
    monthly_counts = {wp: months.value_counts().sort_index() for wp, months in wtdays_dict_monthly.items()}

    # Criando o índice de todos os meses
    all_months = pd.Index(range(1, 13))

    # Contagem de ocorrências por mês para cada padrão
    monthly_counts = {wp: months.value_counts().reindex(all_months, fill_value=0) for wp, months in wtdays_dict_monthly.items()}
    monthly_counts = {wp: counts.rename(lambda x: pd.Timestamp(2023, x, 1).strftime('%b')) for wp, counts in monthly_counts.items()}


    tags = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)']

    fig = plt.figure(figsize=(10, 15))
    rows = m.ceil( ncenters / 2.0) # arredonda para cima
    cols = 2 if ncenters > 1 else 1 
    gs = gridspec.GridSpec(rows,cols, figure=fig,wspace=0.1, hspace=0.3)

    max_count = max(max(counts) for counts in monthly_counts.values())

    for i, (wp, counts) in enumerate(monthly_counts.items()):

        ax = fig.add_subplot(gs[i // cols, i % cols])
        
        gl = ax.grid(zorder=0)
        ax.bar(counts.index, counts, color='#0001C2', zorder=15)
        ax.set_ylim(0, max_count+1)
        ax.set_xticklabels(counts.index, rotation=60, fontdict={'fontsize': 13, 'fontweight': 'bold'})
        #ax.tick_params(axis='y', labelsize=13, labelweight='bold')
        plt.rcParams["axes.labelweight"] = "bold"
        ax.set_title(wp, fontsize=16, weight='bold')
        plt.text(0.01, 1.03, tags[i], transform=ax.transAxes, size=16, weight='bold')

    plt.subplots_adjust(left=0.05,right=0.97,bottom=0.05,top=0.97)

    PDIR2 = PDIR +'/figures/'
    plt.savefig(PDIR2+'monthly_counts.png',dpi=300)
    plt.close()


def season_counts(wtdays_dict, PDIR, ncenters):
    month_to_season = {12: 'DJF', 1: 'DJF', 2: 'DJF',
                       3: 'MAM', 4: 'MAM', 5: 'MAM',
                       6: 'JJA', 7: 'JJA', 8: 'JJA',
                       9: 'SON', 10: 'SON', 11: 'SON'}
    
    seasons_order = ['DJF', 'MAM', 'JJA', 'SON']
    
    # First map the dates to their respective months, and then to their respective seasons
    wtdays_dict_seasons = {wp: [month_to_season[date.month] for date in pd.to_datetime(dates)] for wp, dates in wtdays_dict.items()}
    
    # Count the occurrences for each season
    season_counts = {wp: pd.Series(seasons).value_counts().reindex(seasons_order, fill_value=0) for wp, seasons in wtdays_dict_seasons.items()}

    tags = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)', 'n)', 'o)', 'p)', 'q)', 'r)', 's)']

    fig = plt.figure(figsize=(10, ((ncenters/2) *5) +2))

    rows = m.ceil( ncenters / 2.0) # arredonda para cima
    cols = 2 if ncenters > 1 else 1 
    gs = gridspec.GridSpec(rows,cols, figure=fig,wspace=0.1, hspace=0.3)

    max_count = max(max(counts) for counts in season_counts.values())

    for i, (wp, counts) in enumerate(season_counts.items()):

        ax = fig.add_subplot(gs[i // cols, i % cols])
        
        gl = ax.grid(zorder=0)
        ax.bar(counts.index, counts, color='#0001C2', zorder=15)
        ax.set_ylim(0, max_count+1)
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(seasons_order)))) 
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(seasons_order))

        ax.xaxis.set_tick_params(labelsize=14, labelrotation=45)
        ax.yaxis.set_tick_params(labelsize=14)

        for label in ax.get_yticklabels():
            label.set_fontweight('bold')


        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        #ax.set_xticklabels(counts.index, rotation=60, fontdict={'fontsize': 13, 'fontweight': 'bold'})
        plt.rcParams["axes.labelweight"] = "bold"
        ax.set_title(wp, fontsize=16, weight='bold')
        plt.text(0.01, 1.03, tags[i], transform=ax.transAxes, size=16, weight='bold')

    plt.subplots_adjust(left=0.05,right=0.97,bottom=0.05,top=0.97)

    PDIR2 = PDIR +'/figures/'
    plt.savefig(PDIR2+'season_counts.png',dpi=300)
    plt.close()



def combined_figures(PDIR):
    #PDIR = diretorio
    image_files = [PDIR+'figures/'+'sel_point_WP_sudeste.png',
                   PDIR+'figures/'+'hs_serie.png',
                   PDIR+'figures/'+'elbow_idealclusters.png',   
                   PDIR+'figures/'+'monthly_counts.png', 
                   PDIR+'figures/'+'6patterns.png',
                   PDIR+'figures/'+'WP+WAVE.png']
    
    # Crie uma nova figura
    fig = plt.figure(figsize=(35, 35))

    # Posições para cada eixo (esquerda, inferior, largura, altura)
    # Valores são em frações das dimensões da figura
    # Por exemplo, [0, 0.5, 0.5, 0.5] seria a metade superior esquerda da figura

    positions = [[0, 0.6, 0.4, 0.4], 
                 [0, 0.30, 0.4, 0.4], 
                [0.35, 0.6, 0.3, 0.3], 
                [0, 0.6, 0.4, 0.4], 
                [0.0, 0.0, 0.4, 0.4], 
                [0.37, 0, 1, 1]]


    # Para cada imagem
    for i in range(len(image_files)):
        # Carregue a imagem
        img = mpimg.imread(image_files[i])
        
        # Adicione o eixo na posição especificada
        ax = fig.add_axes(positions[i])
        
        # Esconde os eixos
        ax.axis('off')
        
        # Plote a imagem
        ax.imshow(img)

    # Salve a figura
    plt.savefig(PDIR+'figures/'+'combined_image.png', dpi=600)
    plt.close()


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


def save_df_boia_waverys(boia, df_hs_ponto, PDIR):

    PDIR3 = PDIR + '/data/'
    pasta = pathlib.Path(PDIR3)
    pasta.mkdir(parents=True, exist_ok=True)

    df_hs_boia = pd.read_csv(f'../../historico_{boia}.txt', index_col='# Datetime')
    df_hs_boia = df_hs_boia.Wvht
    df_hs_boia.index = pd.to_datetime(df_hs_boia.index)
    # substituir dados -9999 por nan
    df_hs_boia[df_hs_boia == -9999] = np.nan

    df_hs_boia = df_hs_boia.resample('D').mean()
    # rename index to 'time' to match xarray dataset
    df_hs_boia.index.name = 'time'
    df_hs_boia.index = pd.DatetimeIndex(df_hs_boia.index).normalize()
    # convertendo pd.series para dataframe
    df_hs_boia = df_hs_boia.to_frame()
    df_hs_boia.columns = ['OBS']
    df_hs_ponto.rename(columns={'VHM0':'Waverys'}, inplace=True)
    df_hs_ponto.drop(columns=['latitude','longitude'], inplace=True)

    df_total = pd.concat([df_hs_boia['OBS'],df_hs_ponto['Waverys']],axis=1)


    df_total1 = df_total.loc['1993-01-01':'2017-12-31']
    ###############################################################
    df_name = 'df_total_'+boia+'.csv'
    df_total1.to_csv(PDIR3+df_name,sep=';')
    
########################################
per = 99
percentil = per
boia = 'santos'
testenome = 'scaled_'+ str(boia) + '_extreme_' + str(percentil) 
# Diretorio para salvar dados e figuras #
PDIR=f'../../{testenome}/'
pasta = pathlib.Path(PDIR)
ind = 1 

while pasta.exists():
    PDIR=f'../../{testenome}_{ind}/'
    pasta = pathlib.Path(PDIR)
    ind += 1 

pasta.mkdir(parents=True, exist_ok=False)

print('1 - Selecionando ponto no mapa')
coords = get_point_lat_lon(boia,PDIR)

startwtera = '1993-01-01'
endwtera = '2018-01-01'
#startwtwav = '1993-01-01'
#endwtwav = '2018-01-03'
startwtwav = '1993-01-01'
endwtwav = '2018-01-01'
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

save_df_boia_waverys(boia, df_hs_ponto, PDIR)

from compara_boias2 import compara_boias_daora

#compara_boias_daora(boia, PDIR)


# Plotando serie de hs #
print('5 - Plotando serie de hs')
plot_hs_serie(ds_hs_ponto,Fperc95,percentil, PDIR)

# Selecionando variaveis de onda #
print('6 - Selecionando variaveis de onda')
ds_hs = sel_var_onda(ds_wave,'VHM0')
ds_tp = sel_var_onda(ds_wave,'VTPK')
ds_dir = sel_var_onda(ds_wave, 'VMDR')

# Selecionando dias de ondas extremas no ponto #
print('7 - Selecionando dias de ondas extremas no ponto')
#ds_hs_sel, ds_tp_sel, ds_dir_sel = sel_wave_days(wave95, ds_hs, ds_tp, ds_dir)


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

#ickmeans2 = prekneed1(ds_u2_sel,ds_v2_sel,ds_hgt2_sel)
#sseclusters2 = teste_kmeans(clmax, ickmeans2)
#centers2 = plot_knee(clmax,sseclusters2,PDIR)

print(f'Centroids ideais: {centers}')
# Definindo metodos e núcleos #
print('12 - Definindo metodos e núcleos')
method = 'lloyd'
centroids = centers
#centroids = 12
print(f'Metodo: {method}')
print(f'Núcleos: {centroids}')
print('13 - Rodando Kmeans')
# u1, v1, hgt1, cluster_fraction1, lat, lon, mclusters1, temp = camins(centroids,3,100,
#                                                                         ds_u_sel,
#                                                                         ds_v_sel,
#                                                                         ds_hgt_sel,
#                                                                         method)        


u1, v1, hgt1, cluster_fraction1, lat, lon, mclusters1, temp = camins_update(centroids,3,100,
                                                                        ds_u_sel,
                                                                        ds_v_sel,
                                                                        ds_hgt_sel,
                                                                        method, 
                                                                        scaler_type=None, 
                                                                        joint_scaling='True')


#u2, v2, hgt2, cluster_fraction2, lat, lon, mclusters2, temp2 = camins(centroids,3,100,
                                                                        #ds_u2_sel.to_array().squeeze(),
                                                                        #ds_v2_sel.to_array().squeeze(),
                                                                        #ds_hgt2_sel.to_array().squeeze(),
                                                                        #method)



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

plota_clusters_extreme_waves2(ds_hs,ds_tp,ds_dir,wtdays_dict,u1, v1, hgt1, 
                                lat, lon, cluster_fraction1,lon_boia, lat_boia, boia, PDIR)

#wtdays_dict = pega_dias(mclusters1,temp,cluster_fraction1)
wtdays_dict24 = pega_dias24(mclusters1,temp,cluster_fraction1)
wtdays_dict48 = pega_dias48(mclusters1,temp,cluster_fraction1)
save_dict(wtdays_dict24,'wtdays_dict24',PDIR)
save_dict(wtdays_dict48,'wtdays_dict48',PDIR)



u_2days, v_2days, hgt_2days = before_days(wtdays_dict48, ds_u, ds_v, ds_hgt)
u_1day, v_1day, hgt_1day = before_days(wtdays_dict24, ds_u, ds_v, ds_hgt)

plota_clusters_extreme_waves2_teste(ds_hs,ds_tp,ds_dir,wtdays_dict,u1, v1, hgt1, lat, lon, u_1day, v_1day, hgt_1day, 
                                    u_2days, v_2days, hgt_2days, cluster_fraction1,lon_boia, lat_boia, boia,PDIR)



