import numpy as np 
import os
import json
import pathlib
import joblib
import pandas as pd



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