#!/usr/bin/python


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import datetime as dt
import time
import scipy
import pylab
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt


#np.set_printoptions(threshold=np.nan)

def read_stores_data(filepath):
    #following line is not used.
    types = { 'Store': np.dtype(float),
               'StoreType': np.dtype(str),
               'Assortment': np.dtype(str),
               'CompetitionDistance': np.dtype(float),
               'CompetitionOpenSinceMonth': np.dtype(float),
               'CompetitionOpenSinceYear' : np.dtype(float),
               'Promo2' : np.dtype(float),
               'Promo2SinceWeek' : np.dtype(float),
               'Promo2SinceYear' : np.dtype(float),
               'PromoInterval' : np.dtype(str)}

    #for some reason types does not work. It will be more efficient to build 
    #the data frame with types.
    #store = pd.read_csv("../data/store.csv", dtype=types)
    store = pd.read_csv(filepath)
    return store

def encode_feature(store, featureName):
    """ Encode all the unique values of a column into index based values. 
    The values are numeric so we could apply clustering algorithms. """
    unique_mappings = dict(enumerate(pd.unique(store[featureName].ravel())))
    inv_unique_mappings = {v: k for k, v in unique_mappings.items()}
    store[featureName].replace(inv_unique_mappings, inplace=True)
    return store


def transform_feature(store, featureName):
    """ Transforms a column into new columns based on the unique value
    in the column. Al unique value show up as new columns in the data.
    This transformation is helpful for calculating meaningful similarity
    cofficient. Also see encode_feature"""

    unique_values = list(pd.unique(store[featureName]))
    unique_mappings = dict(enumerate(unique_values))
    inv_unique_mappings = {v: k for k, v in unique_mappings.items()}

    features = [featureName + "_" + str(suffix) for suffix in list(unique_mappings.keys())]
    store = store.reindex(columns=features + list(store.columns), fill_value=0)
    for feature, value in zip(features, unique_values):
        store.loc[store[featureName] == value, feature] = 1
    store.drop(featureName, axis=1, inplace=True)
    return store, {featureName: features}

def min_max_scaling(store, featureName):
    """Avoid calling duw to unnecessary copy of store object. Scales the 
    features as per min max scaling."""
    store[featureName] = (store[featureName] - store[featureName].min()) / (store[featureName].max() - store[featureName].min())
    return store

def preprocess_store_data_from_file(filepath):
    return preprocess_store_data(read_stores_data(filepath))

def preprocess_store_data(store):
    transformed_features = {};

    #trnasform the features. Should not be a problem as number of 
    #unique values are small.
    store, transformation = transform_feature(store, 'StoreType')
    transformed_features.update(transformation)
    store, transformation = transform_feature(store, 'Assortment')
    transformed_features.update(transformation)

    #Promo2 is never null. It make sense that the store knows about when its 
    #promotion is going on. When no promotion is going on, following can be 
    #treated as 0
    promo2_related_info_features = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    store.ix[store.Promo2 == 0, promo2_related_info_features] = 0
    store, transformation = transform_feature(store, 'PromoInterval')
    transformed_features.update(transformation)

    store = min_max_scaling(store, ['Promo2SinceWeek', 'Promo2SinceYear'])

    competitionInfoFeatures = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
    idx = store.CompetitionDistance.isnull() == True
    #set the features for when there is no information about competition. It 
    #probably means that no competition exists. if no store is competiting
    #then it is very far is best guess.
    store.ix[idx, 'CompetitionDistance'] = store['CompetitionDistance'].max() * 1.1;
    #assumption is 0 would provide a better separating value for time which has some unique values.
    #store.ix[idx, ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']] = 1;
    store.ix[store.CompetitionOpenSinceMonth.isnull() == True, 'CompetitionOpenSinceMonth'] = 1
    store.ix[store.CompetitionOpenSinceYear.isnull() == True, 'CompetitionOpenSinceYear'] = 1

    #Other columns does not have any special preprocessing related to them so just dump 
    #the values to 0.
    store = store.fillna(0)

    #Infer additional column for time values.
    store['Timestamp'] = store.apply(lambda row: time.mktime(dt.datetime(year=int(row['CompetitionOpenSinceYear']), month=int(row['CompetitionOpenSinceMonth']), day=1).timetuple()), axis=1)
    store.ix[store['Timestamp'] == 1, 'Timestamp'] = store['Timestamp'].max() *1.1
    store.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True)

    store = min_max_scaling(store, ['Timestamp', 'CompetitionDistance'])
    return store, transformed_features

def kmeans_clustering_stores_from_file(k, filepath):
    data = read_stores_data(filepath);
    return kmeans_clustering_stores(k, data)

def kmeans_clustering_stores(k, data):
    data, transformed_features = preprocess_store_data(data)
    imp_features = ['Store','StoreType_0', 'StoreType_1', 'StoreType_2', 'StoreType_3','Assortment_0', 'Assortment_1', 'Assortment_2', 'CompetitionDistance', 'Timestamp']
    selected_data = data[imp_features]
    #drop unimportant features
    #unimportantFeatures = ['Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    #data = data.drop(unimportantFeatures, axis=1)

    storeFeatures = (selected_data[[col for col in selected_data.columns if col != 'Store']]).as_matrix()

    #Get the columns that would help in clusrering
    #km = KMeans(n_clusters=k)
    km = SpectralClustering(n_clusters=k)
    labels = km.fit_predict(storeFeatures)

    keys = selected_data['Store']
    return dict(zip(keys, labels));

def num_clust_with_gap_stat(data, ks, nrefs=20):
    storeFeatures = (data[[col for col in data.columns if col != 'Store']]).as_matrix()
    return _num_clust_with_gap_stat(storeFeatures, ks, nrefs)

def _num_clust_with_gap_stat(storeFeatures, ks, nrefs=20):
    shape = storeFeatures.shape
    tops = storeFeatures.max(axis=0)
    bots = storeFeatures.min(axis=0)
    dists = scipy.matrix(scipy.diag(tops-bots))
    rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
    for i in range(nrefs):
        rands[:,:,i] = rands[:,:,i]*dists + bots[np.newaxis,:]

    gaps = scipy.zeros((len(ks),))
    s_k = scipy.zeros((len(ks),))
    for (i,k) in enumerate(ks):
        print("Clustering for k: ", k),
	km = KMeans(n_clusters=k, max_iter=1000)
        km.fit(storeFeatures, k)
        kmc = km.cluster_centers_
        kml = km.labels_
	disp = scipy.log(sum([distance.euclidean(storeFeatures[m,:],kmc[kml[m],:]) for m in range(shape[0])]))

	refdisps = scipy.zeros((rands.shape[2],))
	for j in range(rands.shape[2]):
            km.fit(rands[:,:,j], k)
            kmc = km.cluster_centers_
            kml = km.labels_
            refdisps[j] = scipy.log(sum([distance.euclidean(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])]))
	s_k[i] = np.std(refdisps) * np.sqrt(1+1/nrefs)
        gaps[i] = scipy.mean(refdisps) - disp
        print("evaluation parameters are s_k: ", s_k[i], " gap value: ", gaps[i])
    num_clust = -1
    for i in range(len(ks)-1):
        if gaps[i] >= (gaps[i+1] + s_k[i+1]):
            num_clust = ks[i]
            break
    return num_clust, gaps, s_k

def num_clust_with_gap_stat_and_col_preset(filepath):
    ks = [i for i in xrange(1, 50)]
    #All features: 'Store','StoreType','Assortment','CompetitionDistance','Timestamp','Promo2','Promo2SinceWeek','Promo2SinceYear', 'PromoInterval'
    presets = [['Store','StoreType_0', 'StoreType_1', 'StoreType_2', 'StoreType_3','Assortment_0', 'Assortment_1', 'Assortment_2', 'CompetitionDistance', 'Timestamp'], 
            ['Store','StoreType_0', 'StoreType_1', 'StoreType_2','Assortment_0', 'Assortment_1', 'Assortment_2','CompetitionDistance'],
            ['Store','StoreType_0', 'StoreType_1', 'StoreType_2','Assortment_0', 'Assortment_1', 'Assortment_2','CompetitionDistance','Timestamp','Promo2','Promo2SinceWeek','Promo2SinceYear', 'PromoInterval_0', 'PromoInterval_1', 'PromoInterval_2', 'PromoInterval_3']];
    data, transformation_feature = preprocess_store_data_from_file(filepath)
    for (i, imp_feature_set) in enumerate(presets):
        print("Running with preset: ", i)
        num_clust, gaps, s_k = num_clust_with_gap_stat(data[imp_feature_set], ks)
        plt.figure()
        plt.scatter(ks, gaps)
        if num_clust != -1:
            plt.axvline(x=num_clust)
        plt.title("Gap Statistic vs Number of clusters")
        plt.xlabel("Number of clusters")
        plt.ylabel("Gap")
        print("saving file: ")
        pylab.savefig("cluster_quality_preset" + str(i))

#Main script.
def main():
    filepath = 'store.csv'
    stores_data, transformed_features = preprocess_store_data_from_file(filepath)
    #num_clust, gap_stat, s_k = num_clust_with_gap_stat(stores_data, [i for i in xrange(1, 50)])
    #print(gap_stat)
    #print(num_clust)
    #print(s_k)
    num_clust_with_gap_stat_and_col_preset(filepath)
    #if num_clust != -1:
    #    storeLabels = kmeans_clustering_stores(num_clust, filepath)
    #else:
    #    print("Cannot decide on number of clusters.")

if __name__ == "__main__":
    main()
