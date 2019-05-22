# -*- coding: utf-8 -*-

from PIL import Image
import numpy
from numpy.random import rand
from random import choice
import random

#the class of cluster
class Cluster(object):
    def __init__(self):
        self.pixels = []
        self.centroid = None
        
    def addPoint(self, pixel):
        self.pixels.append(pixel)
    
    def setNewCentroid(self):
        R = [colour[0] for colour in self.pixels]   #R-channel
        G = [colour[1] for colour in self.pixels]   #G-channel
        B = [colour[2] for colour in self.pixels]   #B-channel
        #calculate the new means as new centroid
        R = sum(R) / len(R)
        G = sum(G) / len(G)
        B = sum(B) / len(B)
        self.centroid = (R, G, B)
        self.pixels = []
        #return self.centroid

#the class of k-means
class kmeans(object):
    def __init__(self, k = 3, max_iterations = 5, min_distance = 1.0, size = 200):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.size = (size, size)

    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size) #Create Thumbnail
        self.pixels = numpy.array(image.getdata(), dtype = numpy.uint8)
        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        randomPixels = random.sample(list(self.pixels), self.k)
        #Random sampling K pixels as the centroids

        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]
        #initialize k clusters
        
        iterations = 0      #the count of iterations

        while self.shouldExit(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            #update the oldClusters
            print(iterations)

            for pixel in self.pixels:
                self.assignClusters(pixel)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        return [cluster.centroid for cluster in self.clusters]

    def calcDistance(self, a, b):
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):
        if self.oldClusters is None:
            return False
        
        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True
        
        if iterations <= self.max_iterations:
            return False
        
        return True
   
    def assignClusters(self, pixel):
        shortest = float('Inf')     #正无穷

        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def showImage(self):
        self.image.show()

    def showCentroidColours(self):
        for cluster in self.clusters:
            R = int(cluster.centroid[0])
            G = int(cluster.centroid[1])
            B = int(cluster.centroid[2])
            image = Image.new("RGB", (200, 200), (R, G, B))
            image.show()

    def showClustering(self):
        localPixels = [None] * len(self.image.getdata())

        for idx, pixel in enumerate(self.pixels):   #return the index and value
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster
                localPixels[idx] = nearest.centroid

        w, h = self.image.size
        
        localPixels = numpy.asarray(localPixels)
        localPixels = localPixels.astype('uint8')
        localPixels = localPixels.reshape((h, w, 3))
        
        colourMap = Image.fromarray(localPixels)
        colourMap.show()

def main():

    image = Image.open("C:\\Users\\WOXIANG\\Desktop\\Lenna.png")
    k = kmeans()
    result = k.run(image)
    print(result)
    
    k.showImage()
    k.showCentroidColours()
    k.showClustering()

if __name__ == "__main__":
    main()


