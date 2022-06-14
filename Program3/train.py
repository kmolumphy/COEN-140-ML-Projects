# Kevin Molumphy - COEN 140 - Program 3

import numpy as np
import pandas as pd
import math
import random
from sklearn.cluster import KMeans
import sys
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score

#~~ Global Definitions ~~
dataLen = 8580  #Defines current data usage, swap to 8580 for final
distances = []

#~~ Function Definitions ~~

def createClusters(data,numClusters):
    kmeans = KMeans(n_clusters = numClusters,random_state=0).fit(data)
    labels = kmeans.labels_
    #trans = kmeans.transform(data)
    return(kmeans)

def makeClusterPointList(labels,clusterNum):
    clusterPoints = []
    for x in range(clusterNum):
        clusterPoints.append([])
        for labelNum in range(len(labels)):
            lab = int(labels[labelNum])
            if lab == x:
                clusterPoints[x].append(labelNum)
    return clusterPoints

def findDistance(row1,row2,numbers,frequencies):
    length = len(numbers[row1])
    freq1 = frequencies[row1]
    freq2 = frequencies[row2]
    dist = 0
    for num in numbers[row1]:
        if num in numbers[row2]:
            #print("Freq Len:",freqLen,"NumLen: ",length)
            inc = num*(freq2[numbers[row2].index(num)])*(freq1[numbers[row1].index(num)])
            dist+=inc

    distance = dist/length
    distances.append(distance)
    return distance

def findDistance2(row1,row2,numbers,frequencies):
    length = len(numbers[row1])
    freq1 = frequencies[row1]
    freq2 = frequencies[row2]
    dist = 0
    for num in numbers[row1]:
        if num in numbers[row2]:
            inc = (freq2[numbers[row2].index(num)])*(freq1[numbers[row1].index(num)])
            dist+=inc
    distance = dist/length
    distances.append(distance)
    return distance

def findNeighbors(pointsInCluster,rowNum,EPS,numbers,frequencies):
    neighbors = []
    distances = []
    numInCluster = len(pointsInCluster)
    for point in pointsInCluster:
        if rowNum == point:
            break
        dist = findDistance2(rowNum,point,numbers,frequencies)
        distances.append(dist)
        if dist >= EPS:
            #print(dist,EPS,len(pointsInCluster))
            neighbors.append(point)
    #print("RowNum:",rowNum," NumInCluster: ",numInCluster,"Distances: ",distances)

    return neighbors

def DBSCAN(clusterPoints,labels,EPS,MinPts,numbers,frequencies):
    numCluster = 0
    allNeighbors = []
    typeLabels = [0]*dataLen  #defines each rows label as 0=core,1=border,2=noise
    for rowNum in range(dataLen):  #Assign points core and border to cluster, assign noise to k+1 or closest cluster
        pointsInCluster = clusterPoints[labels[rowNum]]
        neighbors = findNeighbors(pointsInCluster,rowNum,EPS,numbers,frequencies)
        allNeighbors.append(neighbors)
        if len(neighbors) >= MinPts:
            typeLabels[rowNum] = 0  #Core Point
        else:
            typeLabels[rowNum] = 2  #Noise Point

    for rowNum in range(dataLen):  #Sets label to border if any neighbors are core points
        if typeLabels[rowNum] == 2:
            neigh = []
            for neighbor in allNeighbors[rowNum]:   
                neigh.append(typeLabels[neighbor])  
                if typeLabels[neighbor] == 0 and typeLabels[rowNum] == 2:
                    typeLabels[rowNum] = 1  #Border Point
                    break
            #print("RowNum: ",rowNum,"Neighbors: ",neigh)

    for points in clusterPoints:  #Prints point Labels
        types = []
        for point in points:
            types.append(typeLabels[point])
        print(types)

    return typeLabels


#~~ Function Calls ~~

listOfNums = []

data = pd.read_csv("train.dat",sep = "\t",header = None)
numRows = data.shape[0]

#turns data from strings to ints

data = data.to_numpy()
count = 0
for y in range(numRows):
    row = data[y]
    count +=1
    nums = row[0].split(' ')
    for x in range(len(nums)):
        #print("Row: ",count," Col: ",x)
        nums[x] = int(nums[x])  
      
    listOfNums.append(nums)

newData = np.asarray(listOfNums,dtype=object)


practiceData = newData[:dataLen]
numbers = []
frequencies = []
for row in practiceData:
    number  = row[::2]
    frequency = row[1::2]
    numbers.append(number)
    frequencies.append(frequency)

#Creates 3 Lists; indptr,indices,data to input for csr_matrix

indptr = [0]
indices = []
data = []
uniqueData = []
for row in numbers:
    for value in row:
        if value not in uniqueData:
            uniqueData.append(value)
        index = uniqueData.index(value)
        indices.append(index)
        data.append(value)
    indptr.append(len(indices))

csr = csr_matrix((data, indices, indptr), dtype=int).toarray()

#Change EPS between 4 and 110



EPS = .8
MinPts = 15
numClusters = 1
startClusters = 120         #Greater or Less than 100?
incrementClusters = 200

types = []
allLabels = []

for i in range(numClusters):
    cluster = startClusters+(i*incrementClusters)
    kmeans = createClusters(csr,cluster)
    labels = kmeans.labels_
    clusterPoints = makeClusterPointList(labels,cluster)
    typeLabels = DBSCAN(clusterPoints,labels,EPS,MinPts,numbers,frequencies)
    clusterCenters = kmeans.cluster_centers_
    types.append(typeLabels)
    allLabels.append(labels)

print(np.percentile(distances,10))
print(np.percentile(distances,90))

file_path = 'ans.txt'
sys.stdout = open(file_path, "w")
loop = types[0]
loopLabels = allLabels[0]
for x in range(len(loop)):
    if loop[x] == 2:
        print(121)
    else:
        print(loopLabels[x])