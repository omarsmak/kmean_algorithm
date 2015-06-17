# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 00:08:52 2015

@author: omaral-safi
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq as vq 

class KMean:
    def __init__(self):
        self.MAX_ITERATIONS = 10
        k = 3
        N = 200
        c, data = self.generateRandomInstances(N, k)
        
        print("Initial Data N=100, k=3:")
        self.plotData(c, data)
        
        print("After running k-means Algorithm N=100, k=3 with Euclidean Distance set:")
        self.plotData(self.kmeans(data, k), data)
        
        print("After running k-means Algorithm N=100, k=3 with Distance function set:")
        self.plotData(self.kmeans(data, k, True), data)
        
        irisData = self.loadData("iris.data")
        print("Initial Iris Data Set (Note, for drawing simplicity we considered the first 2 features 'sepal length and sepal width) loaded")
        
        print("Iris Dataset after running k-means Algorithm k=3 with Euclidean Distance set:")        
        self.plotData(self.kmeans(irisData, k)[:,:2], irisData[:,:2])        
        
        
        print("Iris Dataset after running k-means Algorithm k=3 with Distance function set:")        
        self.plotData(self.kmeans(irisData, k, True)[:,:2], irisData[:,:2])        
    
        
    def plotData(self, c, tests):
        #Plot data to 2D scartter
        #use vq() to assign each sample to a cluster
        assignment,cdist = vq.vq(tests,c)
        plt.scatter(tests[:,0], tests[:,1], c=assignment)
        plt.scatter(c[:,0], c[:,1],s=80, marker='v', c='red')
        plt.show()
        
        
    #Generate random instances, the function takes N (number of instances) and
    # k centroids. It calcautes each instaces to the corresponds normal disrbution
    # N(k[i], sqr(s))
    def generateRandomInstances(self, N, k):
        #we assume N would be distrubted equally among k
        n = float(N)/k
        c = []
        #Generate a random StandardDeviation (s)
        s = np.random.uniform(0.05, 0.5)
        
        results = []
            
        #Generate random centroids (mean) according to the number of k
        for i in range(k):
            c.append((random.uniform(-1, 1), random.uniform(-1, 1)))
            meanResults = []
            while len(meanResults) < n:
                a, b = np.array([np.random.normal(c[i][0], s), np.random.normal(c[i][1], s)])
                #Continue drawing points
                if abs(a) < 1 and abs(b) < 1:
                    meanResults.append([a,b])
            results.extend(meanResults)
            
        results = np.array(results)[:N]
        c = np.array(c)
        
        return (c, results)
        

    # K-Means is an algorithm that takes in a dataset and a constant
    # k and returns k centroids    
    def kmeans(self, data, k, d=False):
        # Initialize centroids randomly
        centroids = data[np.random.choice(range(data.shape[0]),k,replace=False),:]
        
        # Initialize book keeping vars.
        iterations = 0
        oldCentroids = None
        
        while not self.shouldStop(oldCentroids, centroids, iterations):
            # Save old centroids for convergence test. Book keeping.
            oldCentroids = centroids
            iterations += 1          
                
            if(d == False):
                #Distance not set, calculate the Euclidean Distance
                euclideanDistance = np.sqrt(np.sum((centroids[:,np.newaxis,:]-data)**2, axis=2))
            
            else :
                #Distance is set, use the provided deistance instead which is different from Euclidean Distance
                euclideanDistance = np.sum((centroids[:,np.newaxis,:]-data)**2, axis=2)
                        
            
            #Find the closest instance to the center E-Step
            closest = np.argmin(euclideanDistance, axis=0)
            
            #Update the clusetr center M-Step
            for i in range(k):
                centroids[i,:] = data[closest==i, :].mean(axis=0)
                
        
        return centroids
        
        
        
    # Returns True or False if k-means is done. K-means terminates either
    # because it has run a maximum number of iterations OR the centroids
    # stop changing.    
    def shouldStop(self, oldCentroids, centroids, iterations):
        if iterations > self.MAX_ITERATIONS: return True
        return np.array_equal(oldCentroids, centroids)
        
        
        
    def loadData(self, fileName):
        dataset = np.genfromtxt(fileName, delimiter=",", usecols=[0,1,2,3])
        return dataset
        
        
        
        
def main():
    KMean()
    
if __name__ == '__main__': main()