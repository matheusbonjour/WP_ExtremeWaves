import numpy as np





def pega_dias(mclusters, temp, cluster_fraction):
    tempo_days = temp.values 

    ncenters = len(cluster_fraction)

    regimes = []

    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
    ndays = len(mclusters.labels_)
    wt_days = np.zeros((ncenters,ndays)) 
    for lb in range(len(wt_days)):
        wt_days[lb] = mclusters.labels_==lb


    wt_index = []
    for j in range(ncenters):

        ind = [i for i, x in enumerate(wt_days[j]) if x]
        wt_index.append(ind)


    wt_data = []

    for dx in range(ncenters):
        wtd = tempo_days[wt_index[dx]]
        wt_data.append(wtd)


    wt_dict = dict(zip(regimes,wt_data))


    return wt_dict



def pos24h(data):
    #data = datetime.strptime(data_string, '%Y-%m-%d %H:%M:%S')
    um_dia = np.timedelta64(1, 'D')
    
    nova_data = data - um_dia
    
    return nova_data

def pos48h(data):
    
    dois_dias = np.timedelta64(2, 'D')
    
    nova_data = data - dois_dias
    
    return nova_data


def pos72h(data):
        
    tres_dias = np.timedelta64(3, 'D')
        
    nova_data = data - tres_dias
        
    return nova_data

def pos96h(data):

    quatro_dias = np.timedelta64(4, 'D')

    nova_data = data - quatro_dias

    return nova_data


def pega_dias24(mclusters, temp, cluster_fraction):
    tempo_days = temp.values 

    ncenters = len(cluster_fraction)
    
    regimes = []

    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
    ndays = len(mclusters.labels_)
    wt_days = np.zeros((ncenters,ndays)) 
    for lb in range(len(wt_days)):
        wt_days[lb] = mclusters.labels_==lb


    wt_index = []
    for j in range(ncenters):

        ind = [i for i, x in enumerate(wt_days[j]) if x]
        wt_index.append(ind)


    wt_data = []

    for dx in range(ncenters):
        wtd = tempo_days[wt_index[dx]]
        wt_data.append(wtd)

    wt_data24 = []
    for d24 in range(ncenters):

        wtd24 = pos24h(wt_data[d24])
        wt_data24.append(wtd24)

    wt_dict = dict(zip(regimes,wt_data24))


    return wt_dict


def pega_dias48(mclusters, temp, cluster_fraction):
    tempo_days = temp.values 

    ncenters = len(cluster_fraction)
    
    regimes = []

    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
    ndays = len(mclusters.labels_)
    wt_days = np.zeros((ncenters,ndays)) 
    for lb in range(len(wt_days)):
        wt_days[lb] = mclusters.labels_==lb


    wt_index = []
    for j in range(ncenters):

        ind = [i for i, x in enumerate(wt_days[j]) if x]
        wt_index.append(ind)


    wt_data = []

    for dx in range(ncenters):
        wtd = tempo_days[wt_index[dx]]
        wt_data.append(wtd)

    wt_data48 = []
    for d48 in range(ncenters):

        wtd48 = pos48h(wt_data[d48])
        wt_data48.append(wtd48)

    wt_dict = dict(zip(regimes,wt_data48))


    return wt_dict


def pega_dias72(mclusters, temp, cluster_fraction):

    tempo_days = temp.values 

    ncenters = len(cluster_fraction)
    
    regimes = []

    for rg in range(ncenters):
        rgstr = 'WP'+str(rg+1)
        regimes.append(rgstr) 
    ndays = len(mclusters.labels_)
    wt_days = np.zeros((ncenters,ndays)) 
    for lb in range(len(wt_days)):
        wt_days[lb] = mclusters.labels_==lb


    wt_index = []
    for j in range(ncenters):

        ind = [i for i, x in enumerate(wt_days[j]) if x]
        wt_index.append(ind)


    wt_data = []

    for dx in range(ncenters):
        wtd = tempo_days[wt_index[dx]]
        wt_data.append(wtd)

    wt_data72 = []
    for d72 in range(ncenters):

        wtd72 = pos72h(wt_data[d72])
        wt_data72.append(wtd72)

    wt_dict = dict(zip(regimes,wt_data72))


    return wt_dict

