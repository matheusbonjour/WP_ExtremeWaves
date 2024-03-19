import math as m
import numpy as np 
import pandas as pd
import pathlib
import string
import sys
import os
import warnings

import cartopy, cartopy.crs as ccrs 
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import AxesGrid
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

from KneeLocWP import prekneed1, teste_kmeans, plot_knee
from GetDays import pega_dias, pega_dias24, pega_dias48
from ProcessWP import convert_vel, vel_conv, sel_wave_days



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
    fig = plt.figure(figsize=((10), (2*ncenters)+2))
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
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Altura Geopotencial (m)',labelpad=-3,fontsize=16,fontweight='bold') 
    plt.subplots_adjust(left=-0.10,right=0.97,bottom=0.05,top=0.96)
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

def plot_contourf_before(ax, lon, lat, data, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia):
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
    title = 'Campo médio de onda: {}'.format(regimes[j])
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
    title = '24h antes {}'.format(regimes[j])
    ax.set_title(title, fontsize=plt.rcParams['font.size']*2, fontweight='bold')
    plt.text(-0.05, 1, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')

def insert_title48(ax, tags, i, j, cluster_fraction2, regimes):
    title = '48h antes {}'.format(regimes[j])
    ax.set_title(title, fontsize=plt.rcParams['font.size']*2, fontweight='bold')
    plt.text(-0.05, 1, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')


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



def plota_clusters_extreme_waves_before(ds_hs,ds_tp,ds_dir,wtdays_dict1,u_kmeans1, v_kmeans1, hgt_kmeans1, lat, lon, u_1day, v_1day, hgt_1day, 
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
            csa = plot_contourf_before(ax, lon, lat, hgt_2days, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_2days, v_2days, j, sp)
            insert_title48(ax, tags, i, j, cluster_fraction2, regimes)
        
        # 
        elif i % cols == 1:
            j = i // cols
            cmap='RdBu_r'
            csa = plot_contourf_before(ax, lon, lat, hgt_1day, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_1day, v_1day, j, sp)
            insert_title24(ax, tags, i, j, cluster_fraction2, regimes)

        elif i % cols == 2:
            j = i // cols
            cmap='RdBu_r'
            csa = plot_contourf_before(ax, lon, lat, hgt_kmeans1, j, levels, cmap, ncenters, i, cols, lon_boia, lat_boia)
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
    cba.ax.tick_params(labelsize=15)
    cba.set_label('Altura Geopotencial (m)',labelpad=0,fontsize=18,fontweight='bold') 

    cbw = fig.colorbar(csw, cax=cbar_ax_wav, shrink=0.8, aspect=20, orientation='horizontal') 
    cbw.ax.tick_params(labelsize=15)
    cbw.set_label('Altura Significativa de Onda (m)',labelpad=0,fontsize=20,fontweight='bold')

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




    # 
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



def plot_wave_rose(ax, hs_data, dir_data, num_dirs, num_bins, norm, tags, i):
    #print(hs_data)
    #print(dir_data)
    cmap1=cmap_wave()
    dir_rad = np.radians(dir_data)  # Direções já estão em graus, convertendo para radianos
    data_minhs = np.nanmin(hs_data)
    data_maxhs = np.nanmax(hs_data)
    intervalhs = 0.5
    data_minhs = round(data_minhs,1)
    data_maxhs = round(data_maxhs,1)
    levelshs = np.arange(data_minhs,data_maxhs,intervalhs)
    #boundaries = np.arange(data_minhs, data_maxhs, 0.5)
    #boundaries = np.arange(data_minhs, data_maxhs + 0.5, 0.5)
    #norm = BoundaryNorm(boundaries, cmap1.N, clip=True)
    hs_bins = np.linspace(hs_data.min(), hs_data.max(), num_bins)  # Cria os bins para altura da onda
    dir_bins = np.linspace(0, 2 * np.pi, num_dirs + 1)  # Cria os bins para direção


    hist, _, _ = np.histogram2d(dir_rad, hs_data, bins=[dir_bins, hs_bins])  # Histograma de frequência
    hist_hs, _, _ = np.histogram2d(dir_rad, hs_data, bins=[dir_bins, hs_bins], weights=hs_data)  # Histograma ponderado por HS
    
    average_hs = np.divide(hist_hs.sum(axis=1), hist.sum(axis=1), out=np.zeros_like(hist_hs.sum(axis=1)), where=hist.sum(axis=1)!=0)

    #if norm is None:
        #norm = Normalize(vmin=hs_data.min(), vmax=hs_data.max())

    theta = dir_bins[:-1] + (dir_bins[1] - dir_bins[0]) / 2  # Calcula o ângulo médio para cada barra

    # Use a norma e o mapa de cores para colorir as barras com base na altura média da onda
    # = [cmapz(norm(value)) for value in average_hs]

    colors = [cmap1(norm(value)) for value in average_hs]

    bars = ax.bar(theta, hist.sum(axis=1), width=(2 * np.pi / num_dirs), bottom=0.0, edgecolor='black')

    # Definir a cor de cada barra
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    sm = ScalarMappable(cmap=cmap1, norm=norm)
    sm.set_array([])  # Isso é necessário porque não estamos usando a ScalarMappable diretamente em um imshow ou similar

    # Cria e adiciona a colorbar para o gráfico
    #plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)

    # Ajuste as marcas de orientação para N-S-E-W
    ax.set_theta_zero_location('N')  # Define o zero (0 radianos) para o norte
    ax.set_theta_direction(-1)  # Define a direção do aumento do ângulo para o horário

        # Ajustando as marcas de orientação para N-S-E-W e aumentando o tamanho da fonte
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=14)

    

    #ax.set_yticklabels(['0', '2', '4', '6', '8', '10', '12', '14', '16'], fontsize=12, weight='bold')
    #ax.tick_params(axis='y', which='major', labelsize=12, labelweight='bold')

    plt.text(-0.05, 1.01, tags[i], 
        transform=ax.transAxes, 
        va='bottom', 
        fontsize=plt.rcParams['font.size']*3, 
        fontweight='bold')


    # Ajustando o tamanho dos ticks e das labels dos eixos
    ax.tick_params(axis='both', which='major', labelsize=14, width=3)

    return sm


def plota_clusters_extreme_wave_rose(ds_hs,ds_tp,ds_dir,wtdays_dict1,u_kmeans1, v_kmeans1, hgt_kmeans1, 
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
    latmin = lat.min()#.values
    latmax = lat.max()#.values
    lonmin = lon.min()#.values
    lonmax = lon.max()#.values
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
    cols = 3

    fig = plt.figure(figsize=((18.5), (ncenters*7)-2))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=-0.05, hspace=0.1)

    
    
    #rows = 2 if ncenters == 4 else 3

    for i in range(rows*cols):
        #ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
        col = i % cols
        row = i // cols

        if col == 0:
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            j = row
            cmap='RdBu_r'
            csa = plot_contourf(ax, lon, lat, hgt_kmeans1, j, levels, cmap, ncenters, lon_boia, lat_boia)
            ww = plot_quiver(ax, lon, lat, u_kmeans1, v_kmeans1, j, sp)
            insert_title(ax, tags, i, j, cluster_fraction2, regimes)
        
        elif col == 1: 
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            j = row
            
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


        # Coluna 1 - Waves 
        else:
            ax = fig.add_subplot(gs[row, col], polar=True)
            # Lógica para plotar a rosa de ondas
            ds_hs_point = ds_hs.interp(latitude=lat_boia, longitude=lon_boia)
            ds_tp_point = ds_tp.interp(latitude=lat_boia, longitude=lon_boia)
            ds_dir_point = ds_dir.interp(latitude=lat_boia, longitude=lon_boia)
            ds_hs_sel, ds_tp_sel, ds_dir_sel = sel_wave_days(wt_days, ds_hs_point, ds_tp_point, ds_dir_point)
            sm = plot_wave_rose(ax, ds_hs_sel, ds_dir_sel, 16, 10, norm, tags, i)




    cbar_ax_atm= fig.add_axes([0.05, 0.025, 0.420, 0.02])
    cbar_ax_wav= fig.add_axes([0.535, 0.025, 0.420, 0.02])
    #cbar_ax_ros = fig.add_axes([0.70, 0.025, 0.300, 0.02])

    cba = fig.colorbar(csa, cax=cbar_ax_atm, shrink=0.7, aspect=18, orientation='horizontal')
    cba.ax.tick_params(labelsize=15, width=3)
    cba.set_label('Altura Geopotencial (m)',labelpad=0,fontsize=18,fontweight='bold') 

    cbw = fig.colorbar(csw, cax=cbar_ax_wav, shrink=0.7, aspect=18, orientation='horizontal') 
    cbw.ax.tick_params(labelsize=15, width=3)
    cbw.set_label('Altura Significativa de Onda (m)',labelpad=0,fontsize=20,fontweight='bold')

    #cbr = fig.colorbar(sm, cax=cbar_ax_ros, shrink=0.7, aspect=18, orientation='horizontal')
    #cbr.ax.tick_params(labelsize=15, width=3)
    #cbr.set_label('Altura Significativa de Onda (m)',labelpad=0,fontsize=20,fontweight='bold')
    

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



def monthly_counts(wtdays_dict, PDIR, ncenters):
    wtdays_dict_monthly = {}
    for wp, dates in wtdays_dict.items():
        dates = pd.to_datetime(dates)
        wtdays_dict_monthly[wp] = dates.month
    monthly_counts = {wp: months.value_counts().sort_index() for wp, months in wtdays_dict_monthly.items()}

    
    all_months = pd.Index(range(1, 13))

    
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