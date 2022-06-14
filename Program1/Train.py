from re import I
import numpy as np
import pandas as pd
import time
import sys
 


startTime = time.time()

trainingRows = 14438

#amountOfTrainRows = (int)(trainingRows/2) #Picks amount of rows to test on
#amountOfTestRows = amountOfTrainRows

amountOfTrainRows = 40
amountOfTestRows = 20

traindf = pd.read_csv("train.dat",sep = "\t",header = None)

traindf = traindf.to_numpy()

df = traindf#[:amountOfTrainRows]

trainingSentences = df[:,1]
trainingLabels = df[:,0]






minLetters = 3
uniquewords = []
def createListofUniqueWords(sentences):
    sparseMatrix = []
    locInMatrix = 0
    for sentence in sentences:  #Loops through each line
        sentence = ''.join(ch for ch in sentence if ch.isalnum() or ch == ' ')
        sentence = sentence.split()

        sparseMatrix.append([])
        for num in range(len(sentence)):  #Loops and creates list of unique words
            if sentence[num] not in uniquewords and len(sentence[num]) > minLetters:
                uniquewords.append(sentence[num])
                sparseMatrix[locInMatrix].append(uniquewords.index(sentence[num]))  #Creates matrix of the location of each word in the uniquewords list
            if sentence[num] in uniquewords:
                sparseMatrix[locInMatrix].append(uniquewords.index(sentence[num]))
        locInMatrix = locInMatrix + 1
    matrix = np.asarray(sparseMatrix,dtype = object)
    return(matrix)


def createTestMatrix(sentences):
    sparseMatrix = []
    locInMatrix = 0
    for sentence in sentences:
        sentence = ''.join(ch for ch in sentence if ch.isalnum() or ch == ' ')
        sentence = sentence.split()
        sparseMatrix.append([])
        length = range(len(sentence))
        for num in length:  #Loops and creates list of unique words
            if sentence[num] in uniquewords:
                sparseMatrix[locInMatrix].append(uniquewords.index(sentence[num]))
        locInMatrix = locInMatrix + 1
    matrix = np.asarray(sparseMatrix,dtype = object)
    return(matrix)


trainMatrix = createListofUniqueWords(trainingSentences) #runs with sentences from training df
trainingLabels = np.asarray(trainingLabels,dtype = object) #numpy matrix of labels

lab = [0,0,0,0,0]
for x in trainingLabels:
    lab[x-1] = lab[x-1] + 1
#print("Labels",lab)  #Prints distribution of labels in training set


#Creating Test DF
testdf = pd.read_csv("test.dat", sep = "\t", header = None)

testdf = testdf.to_numpy()
testdf = testdf#[:amountOfTestRows]
testSentences = testdf[:,0]
testMatrix = createTestMatrix(testSentences) # runs with sentences from test df



#Create part of training set as test to test accuracy
startingRow = amountOfTrainRows
practicedf = traindf[startingRow:startingRow+amountOfTestRows]
practiceSentences = practicedf[:,1]
practiceLabels = practicedf[:,0]
practiceMatrix = createTestMatrix(practiceSentences)





Words = time.time()
curtime = Words-startTime
timePerRow = curtime/(amountOfTestRows+amountOfTrainRows)
estimateRows = np.size(testdf)+np.size(traindf)
#print("Time to create matrices: ",curtime)
#print("Estimated time per row is ",timePerRow)
#print("Estimated time for: ",estimateRows," rows is: ", estimateRows*timePerRow," Minutes: ",(estimateRows*timePerRow)/60)

#Testing
# 
#for each sentence in new set
    #check surrounding k parameters to decide which one to group it in?


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def calculate1(testWords,trainWords):  #Makes list of how many times each word shows up in train set of words
    #print("Test:",testWords)
    #print("Train:",trainWords)
    distance = len(testWords)
    same = [0]*len(testWords)
    for num in range(len(testWords)):
        same[num] = trainWords.count(testWords[num])
    distance = distance-sum(same)
    return distance

    

def calculate2(testWords,trainWords):  #If word is in training word set subtract by 1 from amount of words for distance
    #print("Test:",testWords)
    #print("Train:",trainWords)
    distance = len(testWords)
    for num in testWords:
        if num in trainWords:
            distance = distance-1
            #print(distance)
    return distance
    

def calculate3(testWords,trainWords): #Minkowski
    distance = len(testWords)
    testBin = [0]*len(testWords)
    for num in range(len(testWords)):
        testBin[num] = trainWords.count(testWords[num])
    
    trainBin = [0]*len(trainWords)
    for num in range(len(trainWords)):
        trainBin[num] = testWords.count(trainWords[num])
    return distance

    




def findLabel(labels,distances):
    #print("Labels: ",labels)
    #print("Distances: ",distances)
    labelWeights = [0,0,0,0,0]
    for x in range(len(labels)): 
        if distances[x] != 0:
            weight = 1/(distances[x]**2)  #weights are 1/d^2
            labelWeights[labels[x]-1] = labelWeights[labels[x]-1] + weight  #increment labelweights by correct labels weight
    #print("LabelWeights: ",labelWeights)
    return(labelWeights.index(max(labelWeights))+1)

def findClosestLabel(words,k):
    kClosest = list(range(1000000,1000000+k))  #k closest distances
    kClosestIndexes = list(range(1000000,1000000+k))  #indixes of k closest distances
    for numRow in range(len(trainMatrix)):

        distance = calculate2(words,trainMatrix[numRow]) #distance from test point to row in training set
        if distance < max(kClosest):  #If distance is closer than largest distance
            biggest = kClosest.index(max(kClosest))  
            kClosest[biggest] = distance  #replace big distance with small distance
            kClosestIndexes[biggest] = trainingLabels[numRow]  #Replace changed index with correct index
    return(findLabel(kClosestIndexes,kClosest))
    


def testKValues(numKs): #Tests
    estimateLabels = []  #Creates list of lists to be made for estimated labels
    for x in range(len(numKs)):
        estimateLabels.append([])
        kVal = numKs[x]  #assigns kVal
        Labels = list(range(0,np.size(testdf)))  #Creates list of size testdf for all rows
        for numRow in range(len(testMatrix)):
            label = findClosestLabel(testMatrix[numRow],kVal)  #Runs and finds estimate label for row
            Labels[numRow] = label
        estimateLabels[x] = Labels
    return(estimateLabels)

kValues = [16]
#print("KValues:",kValues)
estimates = testKValues(kValues)  #runs with kValues

#print("Estimates: ",estimates)


file_path = 'ans.txt'
sys.stdout = open(file_path, "w")
for x in estimates[0]:
    print(x)


#print("Reals: ",practiceLabels)


#Finds accuracy #NOT F1-Score
#for y in range(len(kValues)):   
#    wrong = 0
#    right = 0
#    counter = 0
#    for x in estimates[y]:
#        if(x == practiceLabels[counter]):
#            right = right+1
#        else:
#            wrong = wrong+1
#        #print("X: ",x," P: ",practiceLabels[counter])
#        #print("Kvalue: ",kValues[y]," Accuracy: ",right/(right+wrong))
#        counter = counter +1
#    print("Kvalue: ",kValues[y]," Accuracy: ",right/(right+wrong))
    


#endTime = time.time()
#print("Time to run: ",endTime-startTime)