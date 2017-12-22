# -*- coding: utf-8 -*-
"""
    This code is used to explore the Intel Lab Data

    Author: Linhai Li
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt #has problem, need to write the median filter myself
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

#Define a median filter to remove spikes
def nanmedfilt(x,k=3):
    assert k%2==1 #the window size has to be odd
    
    numel = len(x)
    half_k = (k-1)//2
    
    y = np.zeros((k, numel))       
    
    for j in range(k):
        y[j,half_k:-half_k] = x[j:numel-k+j+1]
        
    y[:,:half_k] = np.tile(x[:half_k], (k,1))
    y[:,-half_k:] = np.tile(x[-half_k:],(k,1))
        
    return np.nanmedian(y, axis=0)

#Define a function that better control the appearance of the subplots
plt.rcParams["font.family"] = "Times New Roman"
def iSubplot(nrows=1,ncols=1,Gap=(0.05,0.05),Min=(0.05,0.05),Max=(0.98,0.98),sharex=False,sharey=False,xscale='linear',yscale='linear'):
    
    fig = plt.gcf()
    fig.clf()
    
    _, axs = plt.subplots(nrows,ncols,sharex=sharex,sharey=sharey,squeeze=False,num=fig.number)
    
    ind_width = (Max[0] - Min[0] - (ncols-1)*Gap[0])/ncols
    ind_height = (Max[1] - Min[1] - (nrows-1)*Gap[1])/nrows
                 
    for nr in range(nrows):
        for nc in range(ncols):
            pos = axs[nr,nc].get_position()
            pos.x0 = nc*ind_width + Min[0] + nc*Gap[0]
            pos.y0 = (nrows-nr-1)*ind_height + Min[1] + (nrows-nr-1)*Gap[1]
            pos.x1 = (nc+1)*ind_width + Min[0] + nc*Gap[0]
            pos.y1 = (nrows-nr)*ind_height + Min[1] + (nrows-nr-1)*Gap[1]
            axs[nr,nc].set_position(pos)  
            #set some default properties
            axs[nr,nc].tick_params(direction='in', which='both', labelsize=10)
            axs[nr,nc].set_xscale(xscale)
            axs[nr,nc].set_yscale(yscale)
    
    return axs

#define a function to plot the data
def plot_data(dgroups, moteID, col1, col2, figNum=1):
    fig = plt.figure(figsize=(7,10), num=figNum)
    axs = iSubplot(nrows=7,ncols=4,Gap=(0.08,0.03),Min=(0.07,0.04),Max=(0.945,0.96)).flatten()
    axs[-1].set_visible(False) #last axes is not used
    
    moteNum = 0
    for m in moteID:    
        ax1 = axs[moteNum]
        pos = ax1.get_position()
        moteNum += 1
        
        gdata = dgroups.get_group(m)
        
        if not isinstance(col1, list):
            ax1.set_prop_cycle('color', ['m', 'c', 'y'])
        else:
            ax1.set_prop_cycle('color', ['k', 'g', 'b'])
            
        lines1 = ax1.plot(gdata['second'], gdata[col1], label=col1)
        ax1.tick_params(labelsize=8)
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left=0)
        ax1.ticklabel_format(style='sci', axis='x')
        ax1.set_xticks([0,1000000,2500000])
        
        if moteNum>23:
            ax1.set_xlabel('Second', fontdict={'fontsize':12})
        if moteNum==13:
            if not isinstance(col1, list):
                ylabel = col1
            else:
                ylabel = ', '.join(map(str, col1))
            ax1.set_ylabel(ylabel, fontdict={'fontsize':12})
        
        ax1.text(0.03,0.8,'mote ' + str(m), transform=ax1.transAxes, ha='left', va='top', fontdict={'fontsize':12, 'fontweight':'bold'})
        
        ax2 = ax1.twinx()
        ax2.set_position([pos.x0, pos.y0, pos.width, pos.height])
        lines2 = ax2.plot(gdata['second'], gdata[col2], 'r', label=col2)
        ax2.tick_params(labelsize=8)
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(left=0)

        if moteNum==16:
            ax2.set_ylabel(col2, fontdict={'fontsize':12})    
        
        if moteNum==27:    
            if not isinstance(col1, list):
                col1 = [col1]
            if not isinstance(col2, list):
                col2 = [col2]    
                
            plt.legend(lines1+lines2, col1+col2, fontsize=12, loc='lower left', bbox_to_anchor=(0,0.965,1,0.05), \
                       mode='expand', ncol=3, handlelength=1, borderaxespad=0.1, edgecolor='k', bbox_transform=fig.transFigure)
        
    plt.show(block=False)
    
    return fig

########Main block code to run###################
if __name__ == "__main__":

    data = pd.read_csv('data.txt', sep='\s+', names=['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage'])
    mote_loc = pd.read_csv('mote_locs.txt', sep='\s+', names=['moteid', 'x', 'y'])
    
    #Process the time vector of the measurements to microseconds for better handling the data
    data['datetime']  = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y-%m-%d %H:%M:%S.%f')
    startTime = data['datetime'].min()
    data['second'] = (data['datetime'] - startTime) / np.timedelta64(1, 's')
    
    #Later, in order to get the actual date and time of measurements, using
    #mDateTime = pd.to_timedelta(data['second'], unit='s') + startTime
                                
    cols = ['temperature', 'humidity', 'voltage', 'light']
    
    data = data.groupby('moteid').apply(lambda x: x.sort_values(['second'])).reset_index(drop=True)
    
    ####Visualize the original data in ascending time for each moteid
    mgroup = data.groupby('moteid')
    
    print('Plotting original data ...')
    
    plt.close('all')
    ##temperature, humidity and voltage
    #Part 1
    fig = plot_data(mgroup, mote_loc['moteid'][:27], cols[:2], cols[2], 1)
    fig.savefig('temp_hum_volt_part_1.tiff', dpi=600, bbox_inches='tight')
    
    #Part 2
    fig = plot_data(mgroup, mote_loc['moteid'][27:], cols[:2], cols[2], 2)
    fig.savefig('temp_hum_volt_part_2.tiff', dpi=600, bbox_inches='tight')
    
    #####light and voltage###############################
    #Part 1
    fig = plot_data(mgroup, mote_loc['moteid'][:27], cols[3], cols[2], 3)
    fig.savefig('light_volt_part_1.tiff', dpi=600, bbox_inches='tight')
    
    #Part 2
    fig = plot_data(mgroup, mote_loc['moteid'][27:], cols[3], cols[2], 4)
    fig.savefig('light_volt_part_2.tiff', dpi=600, bbox_inches='tight')
    
    #########Hypothesis: battery life has significant correlation with the temperature###########
    #define the range for temperature, humidity, voltage, and light, respectively
    drange = [(0,40), (0,100), (2,3), (0,2000)]
     
    columns = ['moteid', 'minTemp', 'minHum', 'minVolt', 'minLight', 'maxTemp', 'maxHum', 'maxVolt', 'maxLight', \
               'avgTemp', 'avgHum', 'avgVolt', 'avgLight', 'stdTemp', 'stdHum', 'stdVolt', 'stdLight', 'lastVolt', 'duration']
    hypoDF =  pd.DataFrame([], columns=columns)
    
    #temperature thresholding have to be done for each sensor separately
    for m in mote_loc['moteid']:
        gdata = mgroup.get_group(m)
        
        #get index for temperature>40; not so easy because there are nan's for temperature
        gtemp = gdata['temperature']
        t_start = np.argwhere(gtemp>40).flatten()
        if len(t_start)==0:
            t_start = -1
        else:
            t_start = t_start[0]        
        t_start = gdata.index[t_start]
        t_first = gdata.index[gdata.index<t_start]
        t_last = gdata.index[gdata.index>=t_start]     
        
        ###Clean the data before calculations
        for i in range(len(cols)):
            #remove the data outside the ranges
            gdata.loc[gdata[cols[i]]>drange[i][1],cols[i]] = np.nan
            gdata.loc[gdata[cols[i]]<drange[i][0],cols[i]] = np.nan
            
            #despike and remove data with temperature>30
            gdata.loc[:,cols[i]] = nanmedfilt(gdata[cols[i]])
            if i<2: #not voltage and light column
                gdata.loc[t_last, cols[i]] = np.nan 
                         
            #further remove the data outside the 1% and 99% percentile
            if i!=2: #not voltage data
                qrange = gdata[cols].quantile([0.01,0.99]) #calculate the 1% and 99% percentile
                gdata.loc[gdata[cols[i]]>qrange.loc[0.99,cols[i]],cols[i]] = np.nan
                gdata.loc[gdata[cols[i]]<qrange.loc[0.01,cols[i]],cols[i]] = np.nan
        
        #Obtain the non-NaN voltage values when temperature went up to 40
        v_index = np.argwhere(np.isnan(gdata.loc[t_first, 'voltage'])==False)
        if len(v_index)<1:
            v_index = t_first[-1]
        else:
            v_index = t_first[v_index[-1]][0]
        
        temp = [[m] + list(gdata.loc[t_first, cols].min().values) + list(gdata.loc[t_first, cols].max().values) + \
                list(gdata.loc[t_first, cols].mean().values) + list(gdata.loc[t_first, cols].std().values) + \
                [gdata.loc[v_index,'voltage']] + [gdata.loc[t_first[-1], 'second']]]
        tempDF = pd.DataFrame(temp, columns=columns, index=[m-1])
        hypoDF = hypoDF.append(tempDF)   
            
        data.loc[gdata.index, gdata.columns] = gdata
               
    ####Visualize the cleaned data in ascending time for each moteid
    mgroup = data.groupby('moteid')
    
    print('Plotting cleaned data ...')
    
    ##temperature, humidity and voltage
    #Part 1
    fig = plot_data(mgroup, mote_loc['moteid'][:27], cols[:2], cols[2], 5)
    fig.savefig('clean_temp_hum_volt_part_1.tiff', dpi=600, bbox_inches='tight')
    
    #Part 2
    fig = plot_data(mgroup, mote_loc['moteid'][27:], cols[:2], cols[2], 6)
    fig.savefig('clean_temp_hum_volt_part_2.tiff', dpi=600, bbox_inches='tight')
    
    #####light and voltage###############################
    #Part 1
    fig = plot_data(mgroup, mote_loc['moteid'][:27], cols[3], cols[2], 7)
    fig.savefig('clean_light_volt_part_1.tiff', dpi=600, bbox_inches='tight')
    
    #Part 2
    fig = plot_data(mgroup, mote_loc['moteid'][27:], cols[3], cols[2], 8)
    fig.savefig('clean_light_volt_part_2.tiff', dpi=600, bbox_inches='tight')
    
    ###########Try to get some relationship and classify the sensors into groups#############
    hypoDF.dropna(inplace=True)
    
    #moteid = 15 seems an outlier
    hypoDF.drop(index=hypoDF[hypoDF['moteid']==15].index, inplace=True)
    
    print('Plotting regression relationships ...')
    
    #make plots to show the a few relationships
    fig = plt.figure(figsize=(6.5, 2.2), num=9)
    axs1 = iSubplot(nrows=1, ncols=3, Gap=(0.12,0.01), Min=(0.065, 0.16), Max=(0.985, 0.95)).flatten()
    
    #Duration of good measurements vs initial voltage
    ax = axs1[0]
    x = hypoDF['maxVolt']
    y = hypoDF['duration']/3600./24
    reg = LinearRegression(n_jobs=5)
    reg.fit(x.values.reshape((-1,1)),y.values.reshape((-1,1)))
    ax.plot(x, y, 'o', markerfacecolor=None)
    px = np.linspace(2,3,100).reshape((-1,1))
    ax.plot(px, reg.predict(px), 'k-', lw=0.5)
    ax.set_xlim(left=2, right=3)
    ax.set_ylim(bottom=0, top=30)
    ax.set_xlabel('Initial voltage (V)', fontdict={'fontsize':12})
    ax.set_ylabel('Duration (days)', fontdict={'fontsize':12})
    ax.tick_params(labelsize=8)
    
    #Drainage rate of the battery vs fluctuation of temperature
    ax = axs1[1]
    x = hypoDF['stdTemp']
    y = (hypoDF['maxVolt']-hypoDF['minVolt'])/(hypoDF['duration']/3600./24)
    reg = LinearRegression(n_jobs=5)
    reg.fit(x.values.reshape((-1,1)),y.values.reshape((-1,1)))
    ax.plot(x, y, 'o', markerfacecolor=None)
    px = np.linspace(0,5,100).reshape((-1,1))
    ax.plot(px, reg.predict(px), 'k-', lw=0.5)
    ax.set_xlim(left=0, right=5)
    ax.set_ylim(bottom=0, top=0.04)
    ax.set_xlabel(r'STD Temperature ($^oC$)', fontdict={'fontsize':12})
    ax.set_ylabel('Drainage rate (V/day)', fontdict={'fontsize':12})
    ax.tick_params(labelsize=8)
    
    #Drainage rate of the battery vs fluctuation of humidity
    ax = axs1[2]
    x = hypoDF['stdHum']
    y = (hypoDF['maxVolt']-hypoDF['minVolt'])/(hypoDF['duration']/3600./24)
    reg = LinearRegression(n_jobs=5)
    reg.fit(x.values.reshape((-1,1)),y.values.reshape((-1,1)))
    ax.plot(x, y, 'o', markerfacecolor=None)
    px = np.linspace(0,10,100).reshape((-1,1))
    ax.plot(px, reg.predict(px), 'k-', lw=0.5)
    ax.set_xlim(left=0, right=10)
    ax.set_ylim(bottom=0, top=0.04)
    ax.set_xlabel(r'STD Humidity (%)', fontdict={'fontsize':12})
    ax.set_ylabel('Drainage rate (V/day)', fontdict={'fontsize':12})
    ax.tick_params(labelsize=8)
    
    plt.show(block=False)
    fig.savefig('example_relationship.tiff', dpi=600, bbox_inches='tight')
    
    ###The following show example of clustering
    feature_cols = columns[1:-2]
    feature_cols = [c for c in feature_cols if 'Volt' not in c] #Remove columns related with voltage (not a environment feature)
    
    print('Determining number of clusters ... ')
    #Check how many clusters by plotting inertia vs n_clusters
    inertia = []
    label = []
    clust = []
    for nclust in range(3,15):
        kmeans = KMeans(n_clusters=nclust, random_state=42, n_jobs=-1).fit(hypoDF[feature_cols])
        inertia.append(kmeans.inertia_)
        label.append(kmeans.labels_)
        clust.append(nclust)
        
    fig = plt.figure(num=10)
    plt.plot(clust, inertia, '-o')
    plt.show(block=False)
    
    #Based on elbow-feature, decide to take 10 clusters
    print('Clustering ...')
    
    clust = 10
    kmeans = KMeans(n_clusters=clust, random_state=42, n_jobs=-1).fit(hypoDF[feature_cols])
    label = kmeans.labels_
    
    print('Plotting clustering results ...')
            
    fig = plt.figure(figsize=(6,3), num=11)
    ax = iSubplot(Min=(0.02, 0.02), Max=(0.98, 0.98)).flatten()[0]
    
    ax.set_prop_cycle('color', ['r', 'g', 'b', 'c', 'm', 'k', 'y', 'purple', 'pink', 'gray'])    
    ax.invert_yaxis()
    ax.invert_xaxis()
    for nclust in range(0,clust):
        idx = np.argwhere(label==nclust).flatten()
        idx = np.in1d(mote_loc['moteid'],hypoDF.iloc[idx]['moteid'])
        locx = mote_loc.loc[idx, 'x']
        locy = mote_loc.loc[idx, 'y']
        ax.plot(locx, locy, 'o')
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show(block=False)
    fig.savefig('clustering_result.tiff', dpi=600, bbox_inches='tight')
    
    print('Finished!')
