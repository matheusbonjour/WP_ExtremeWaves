import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker
import numpy as np
import xarray as xr
import cmocean.cm as cmo
from matplotlib.patches import Patch
import os
import geopandas as gpd
import json
import pathlib
import matplotlib.patheffects as pe
# Definir o diretório de trabalho
#shapefile_path = "../campos_de_producao_jan2023/"
#shapefile_path2 = "../Poligono_Pre_Sal/"

# Definir arquivo shp
#shp = shapefile_path2 + 'Poligono_Pre_Sal.shp'
#shp2 = shapefile_path + 'CAMPOS_DE_PRODUCAO_mar2023.shp'

# Carregar o arquivo NetCDF
#ds = xr.open_dataset("../ETOPO1_Bed_g_gmt4.grd")

# Carregar o shapefile
#gdf = gpd.read_file(shp)
#gdf2 = gpd.read_file(shp2)

class CreateMapBoia:
    def __init__(self, region, buoy, PDIR):
        #self.dataset_path = dataset_path
        self.region = region
        self.buoy = buoy
        self.PDIR = PDIR 
        self.click_count = 0
        self.create_map()

    def create_map(self):

        #ds = xr.open_dataset(self.dataset_path)

        #batimetria = ds["z"]

        if self.region == 'sudeste':
            lon_min, lon_max = -50, -36
            lat_min, lat_max = -32, -15
        elif self.region == 'sul':
            lon_min, lon_max = -60, -45
            lat_min, lat_max = -40, -20
        elif self.region == 'nordeste':
            lon_min, lon_max = -45, -30
            lat_min, lat_max = -20, 5
        elif self.region == 'atlsul':
            lon_min, lon_max = -60, -30
            lat_min, lat_max = -40, -10

        if self.buoy == 'vitoria':
            self.lon = -39.69
            self.lat = -19.92
            self.name = 'Boia Vitória'
            self.lonsantos = -45.03
            self.latsantos = -25.43
            self.namesantos = 'Boia Santos'
            self.lonriograde = -49.83
            self.latriogrande = -31.56
            self.nameriogrande = 'Boia Rio Grande'
        elif self.buoy == 'santos':
            self.lon = -45.03
            self.lat = -25.43
            self.name = 'Boia Santos'
            self.lonvitoria = -39.69
            self.latvitoria = -19.92
            self.namevitoria = 'Boia Vitoria'
            self.lonriograde = -49.83
            self.latriogrande = -31.56
            self.nameriogrande = 'Boia Rio Grande'
        elif self.buoy == 'riogrande':
            self.lon = -49.83
            self.lat = -31.56
            self.name = 'Boia Rio Grande'
            self.lonvitoria = -39.69
            self.latvitoria = -19.92
            self.namevitoria = 'Boia Vitoria'
            self.lonsantos = -45.03
            self.latsantos = -25.43
            self.namesantos = 'Boia Santos'




            

            
        #batimetria_recortada = batimetria.sel(x=slice(lon_min, lon_max), y=slice(lat_min, lat_max))

        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        gl = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()

        #batimetria_recortada = batimetria_recortada * -1
        #levels = np.arange(0, 7000, 150)

        #cf = self.ax.contourf(batimetria_recortada.x, batimetria_recortada.y, batimetria_recortada, levels=levels, cmap=cmo.deep, transform=ccrs.PlateCarree(), vmin=0, vmax=5000)
        #colorbar = plt.colorbar(cf, orientation='horizontal', pad=0.05, shrink=0.7)
        #profundidade = colorbar.ax.set_xlabel('Profundidade (m)', fontsize=15)

        resol = '10m'

        self.ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land']))
        self.ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7))

        # adicionando ilhas 


        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        

        if self.buoy == 'vitoria':

            self.ax.plot(self.lon, self.lat, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lon, self.lat - 0.5, f'{self.name}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

            self.ax.plot(self.lonsantos, self.latsantos, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lonsantos, self.latsantos - 0.5, f'{self.namesantos}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

            self.ax.plot(self.lonriograde, self.latriogrande, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lonriograde, self.latriogrande - 0.5, f'{self.nameriogrande}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

        elif self.buoy == 'santos':

            self.ax.plot(self.lon, self.lat, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lon, self.lat - 0.5, f'{self.name}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

            self.ax.plot(self.lonvitoria, self.latvitoria, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lonvitoria, self.latvitoria- 0.5, f'{self.namevitoria}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

            self.ax.plot(self.lonriograde, self.latriogrande, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lonriograde, self.latriogrande - 0.5, f'{self.nameriogrande}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])
            
        
        elif self.buoy == 'riogrande':

            self.ax.plot(self.lon, self.lat, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lon, self.lat - 0.5, f'{self.name}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

            self.ax.plot(self.lonvitoria, self.latvitoria, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lonvitoria, self.latvitoria- 0.5, f'{self.namevitoria}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

            self.ax.plot(self.lonsantos, self.latsantos, marker='o', markersize=6, color='red', alpha=0.7, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree())
            self.ax.text(self.lonsantos, self.latsantos - 0.5, f'{self.namesantos}', fontsize=12,fontweight='bold', ha='center', va='top', color='black', transform=ccrs.PlateCarree(),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])


                


        file = f'sel_point_WP_{self.region}+{self.buoy}.png'
        PDIR2 = self.PDIR + '/figures/'
        pasta = pathlib.Path(PDIR2)
    
        pasta.mkdir(parents=True, exist_ok=True)

        plt.savefig(PDIR2+file, dpi=300, bbox_inches='tight')
        plt.close()


    
    def get_lat_lon(self):
        lat = self.lat
        lon = self.lon
        coords_dict = {'lat': lat, 'lon': lon}
        PDIR3 = self.PDIR + '/data/'
        pasta = pathlib.Path(PDIR3)
    
        pasta.mkdir(parents=True, exist_ok=True)
    
        with open(PDIR3+'coords_{self.region}.json', 'w') as f:
            json.dump(coords_dict, f)
        print('Coordenadas salvas com sucesso!')
        return coords_dict
    




"""

shapefile_path = "../campos_de_producao_jan2023/"
shapefile_path2 = "../Poligono_Pre_Sal/"
dataset_path = "../ETOPO1_Bed_g_gmt4.grd"


region = 'sudeste'
petro = 's'
# Executar função

mapa = SelPointMap(shapefile_path, shapefile_path2, dataset_path, region , petro)

coords = mapa.get_lat_lon()
"""