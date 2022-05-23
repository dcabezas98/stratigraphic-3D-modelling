EPSILON=1 # Threshold for zero distance

def distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))

def weight(d):
    if d<EPSILON:
        return 1/EPSILON
    else:
        return 1/d


import numpy as np
from collections import Counter

from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    
    def __init__(self, K, min_neighs, metric):
        self.K=K
        self.metric=metric
        self.neigh = NearestNeighbors(n_neighbors=K, metric=self.metric)
        self.min_neighs=min_neighs
        self.X=[]
        self.Y=[]
        self.labels=[]
        
    def fit(self, X, Y):
        self.neigh.fit(X,Y)
        self.X=X
        self.Y=Y
        self.labels=np.unique(Y)
    
    def weights(self, neighbors, distances): # Weights and labels
        weights=np.zeros(len(distances))
        neighbor_labels=[self.Y[i] for i in neighbors]
        count=Counter(neighbor_labels)
        for i in range(len(neighbors)):
            if count[neighbor_labels[i]]>=self.min_neighs:
                weights[i]=weight(distances[i])
        return weights, neighbor_labels
    
    def vote(self, w, l):
        v=np.zeros(len(self.labels))
        for j in range(len(w)):
            v[l[j]]+=w[j]
        return v

    def predict(self,X):
        predictions=np.zeros(len(X), np.int)
        distances,neighborhoods=self.neigh.kneighbors(X,return_distance=True)
        for i in range(len(X)):
            w,l=self.weights(neighborhoods[i],distances[i])
            v=self.vote(w,l) # Results of voting
            predictions[i]=int(np.argmax(v))
        return predictions
    
    def _meaning(self, x):
        return self.predict([x])[0]
        
    def predict_prob(self, X):
        predictions=np.zeros((len(X),len(self.labels)), np.int)
        distances,neighborhoods=self.neigh.kneighbors(X,return_distance=True)
        for i in range(len(X)):
            w,l=self.weights(neighborhoods[i],distances[i])
            v=self.vote(w,l) # Results of voting
            predictions[i]=v
        return predictions
