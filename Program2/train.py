# Kevin Molumphy - COEN 140 - Program 2

import numpy as np
import pandas as pd
import math
import random
from sklearn import tree
import sys

trainingRows = 1566
testingRows = 392


###### making training arrays ######
traindf = pd.read_csv("train.dat",sep = "\t",header = None)

traindf = traindf.to_numpy()

df = traindf #[:amountOfTrainRows]

trainingSentences = df[:,1]
trainingLabels = df[:,0]
trainingLabels=trainingLabels.astype('int')


[placeholder,practiceTestSentences,practiceTrainSentences] = np.split(trainingSentences,[0,1],axis=0)


[placeholder,practiceTestLabels,practiceTrainLabels] = np.split(trainingLabels,[0,1],axis=0)

#practiceTrainSentences = trainingSentences
#practiceTrainLabels = trainingLabels   

################# training Arrays ############



################# testing arrays ############


testdf = pd.read_csv("test.dat",sep = "\t",header = None)

testdf = testdf.to_numpy()

################## testing arrays #############

#print(np.shape(testdf),np.shape(practiceTestSentences))
row = testdf[0]
actualTestDf = []
for row in testdf:
    newRow = row[0]
    actualTestDf.append(newRow)

# Converts list of lists to list of strings




#Function Definitions

def createMatrix(peptides):  #Creates matrix of each peptide and individual residues in it
    matrix = []
    locInMatrix = 0
    for peptide in peptides:  #Loops through each line
        matrix.append([])
        for num in range(len(peptide)):
            matrix[locInMatrix].append(peptide[num])
        locInMatrix = locInMatrix + 1
    retMatrix = np.asarray(matrix,dtype = object)
    return(retMatrix)


def makeUniqueResiduesList(listOfPeptides,maxLength):  #Makes list of unique residues based on kmer lengths and matrices
    listOfResidues = []
    sparseMatrix = []
    nonSparseMatrix = []
    numInSparseMatrix = 0
    numInMatrix = 0
    numResidues = 0
    for numOfPeptides in range(len(listOfPeptides)):
        sparseMatrix.append([])
        peptide = listOfPeptides[numOfPeptides]
        for num in range(len(peptide)-(maxLength-1)):
            numResidues = numResidues + 1
            if peptide[num:num+maxLength] not in listOfResidues:
                listOfResidues.append(peptide[num:num+maxLength])
            sparseMatrix[numInMatrix].append(listOfResidues.index(peptide[num:num+maxLength]))  #Adds location of each residue in unique list
        numInMatrix = numInMatrix + 1

    for numOfPeptides in range(len(listOfPeptides)):  #Loop and create nonSparseMatrix
        peptide = listOfPeptides[numOfPeptides]
        nonSparseMatrix.append([0]*len(listOfResidues))
        row = nonSparseMatrix[numInSparseMatrix]
        for num in range(len(peptide)-(maxLength-1)):
            row[listOfResidues.index(peptide[num:num+maxLength])] = row[listOfResidues.index(peptide[num:num+maxLength])]+1  #Incremnts proper class in non sparse matrix by 1
        numInSparseMatrix = numInSparseMatrix + 1

    returnList = []
    returnList.append(listOfResidues)
    returnList.append(sparseMatrix)
    returnList.append(nonSparseMatrix)
    returnList.append(numResidues)
    return(returnList)
        

def makeTestMatrices(listOfPeptides,maxLength,listOfResidues):
    nonSparseMatrix = []
    numInSparseMatrix = 0
    numIncrements = 0
    for numOfPeptides in range(len(listOfPeptides)):
        peptide = listOfPeptides[numOfPeptides]
        nonSparseMatrix.append([0]*len(listOfResidues))
        row = nonSparseMatrix[numInSparseMatrix]
        for num in range(len(peptide)-(maxLength-1)):
            for letter in range(len(listOfResidues)):
                newLetter = listOfResidues[letter]
                compResidue = peptide[num:num+maxLength]
                if compResidue[0] == newLetter[0]:
                    #print(listOfResidues.index(listOfResidues[letter]),row[listOfResidues.index(listOfResidues[letter])])
                    row[listOfResidues.index(listOfResidues[letter])] = row[listOfResidues.index(listOfResidues[letter])] + 1
            
        numInSparseMatrix = numInSparseMatrix + 1
    #print(nonSparseMatrix)
    return nonSparseMatrix






def diffKMers(trainMatrix,testMatrix,amt):  #Finds amount of residues in each kmers list of unique residues
    lists = []
    for num in range(amt):
        list = makeUniqueResiduesList(trainMatrix,num+1)     
        list.append(makeTestMatrices(testMatrix,num+1,list[0]))
        list.append(num)
        lists.append(list)
    return lists


def findGuesses(guessMatrix):
    guesses = []
    data = np.percentile(guessMatrix,98)
    for num in range(len(guessMatrix)):
        if guessMatrix[num] > data:
            guesses.append(1)
        else:
            guesses.append(-1)

    return(guesses)


def KNN():
    return 0

def decisionTree(trainMatrix,trainValues,testMatrix):#,testValues):
    #print("Train: ",np.shape(trainMatrix),"Test: ",np.shape(testMatrix))
    #print(trainMatrix[0],testMatrix[0])
    X = trainMatrix
    Y = trainValues
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    #print(clf.predict([X[10]])) #Predicts with input values 


    #print(np.shape(trainMatrix),np.shape(testMatrix))
    guesses = []
    for x in range(len(testMatrix)):
        #actual = testValues[x]
        estimate = clf.predict([testMatrix[x]])
        #print("Compare: ",actual,estimate)
        guesses.append(estimate[0])
    return guesses
        
 

def logisticRegression():
    return 0

def naiveBayes(sparse,unique,totalResidues):
    #find probability of a 1 or -1 for each attribute
    probabilityMatrix = [0]*len(unique)
    numRows = len(sparse)

    for peptide in sparse:  # Finds probability of each unique value to be a 1
        #print(peptide)
        for value in peptide:
            probabilityMatrix[value] = probabilityMatrix[value] + 1
    #print(probabilityMatrix)
    guessMatrix = []
    for num in range(numRows):
        guessMatrix.append([])
        totAvgProbability = 0
        peptide = sparse[num]
        numResidues = len(peptide)
        for residue in peptide:
            totAvgProbability = totAvgProbability + (probabilityMatrix[residue]/totalResidues)
        totAvgProbability = totAvgProbability/numResidues
        guessMatrix[num] = totAvgProbability
        #print("Row: ",num," Prob: ",totAvgProbability)

    returns = []
    returns.append(findGuesses(guessMatrix))
    returns.append(probabilityMatrix)
    return(returns)
        
        

# Returns random guesses list
def makeRandom():
    guesses = []
    options = [-1,1]
    for x in range(392):
        guesses.append(random.choice(options))
    return(guesses)



# Finds MCC of estimates compared to actual set
def findMCC(estimateLabels,actualLabels):
    #print("Estimates: ",estimateLabels," Actuals: ",actualLabels)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pos = 1
    neg = -1
    for num in range(len(estimateLabels)):
        if estimateLabels[num] == pos and actualLabels[num] == pos:
            tp = tp + 1
        if estimateLabels[num] == neg and actualLabels[num] == neg:
            tn = tn + 1
        if estimateLabels[num] == pos and actualLabels[num] == neg:
            fp = fp + 1
        if estimateLabels[num] == neg and actualLabels[num] == pos:
            fn = fn + 1
    print("tp: ",tp," tn: ",tn," fp: ",fp," fn: ",fn)
    top = (tp*tn)-(fp*fn)
    bot = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    MCC = top/bot
    return MCC


def extractFeatures(sparse,unique,probMatrix):
    incrementPercent = 20
    #prob matrix is in order of unique matrix, which attributes have which probability
    eliminate = []
    limit = np.percentile(probMatrix,incrementPercent)
    for num in range(len(probMatrix)):
        if probMatrix[num] < limit:
            eliminate.append(unique[num])

    listOfElims = []
    for num in range(len(eliminate)): #goes through length of eliminate
        for row in sparse: #For each row in sparseMatrix
            rowRemove = unique.index(eliminate[num])
            if rowRemove in row: #if each part of eliminate is in row
                #print(row)
                row.remove(rowRemove) #Remove that part from row
                #print(row)
        listOfElims.append(eliminate[num])

    
    for elim in listOfElims:
        print("Removing Element: ",elim, "Num: ",unique.index(elim))
        unique.remove(elim)


def makeNewSparse(sparse,unique):
    newMatrix = []
    numInMatrix = 0
    for numRow in range(len(sparse)):
        newMatrix.append([])
        row = sparse[numRow]
        for residue in row:
            newMatrix[numRow].append(unique.index(residue))
            print(residue)


def extractNonSparseFeatures(trainingMatrix,testingMatix):
    rows = np.shape(trainingMatrix)[0]
    cols = np.shape(trainingMatrix)[1]
    rowImportance = [0]*cols
    for columnNum in range(cols):
        for rowNum in range(rows):
            row = trainingMatrix[rowNum]
            rowImportance[columnNum] = rowImportance[columnNum] + row[columnNum]

    numColsRemoved = 1

    for x in range(numColsRemoved):
        minIndex = rowImportance.index(min(rowImportance))
        trainingMatrix = np.delete(trainingMatrix,minIndex,1)
        testingMatix = np.delete(testingMatix,minIndex,1)

    returnList = []
    returnList.append(trainingMatrix)
    returnList.append(testingMatix)
    return returnList

            



def callClassifier(trainLists,testingLabels):
    scores = []
    for numList in range(len(trainLists)):
        trainList = trainLists[numList]
        unique = trainList[0]
        sparse = trainList[1]
        nonSparse = trainList[2]
        numResidues = trainList[3]
        testNonSparse = trainList[4]
        kmer = trainList[5]



        loops = 4
        for loop in range(loops):
            #print(testNonSparse)
            guesses = decisionTree(nonSparse,testingLabels,testNonSparse)#,practiceTestLabels)
            print(guesses)
            #guesses = naiveBayes(sparse,unique,numResidues)  # Change function to determine which classifier we use
            #res = findMCC(guesses,practiceTestLabels)  # Calls findMCC with guesses as estimates
            #scores.append(res)
            #print("KMer #: ",kmer+1," Loop #: ",loop," Score: ",res)

            returns = extractNonSparseFeatures(nonSparse,testNonSparse)

            nonSparse = returns[0]
            testNonSparse = returns[1]

            #extractFeatures(sparse,unique,guesses[1])
            #makeNewSparse(sparse,unique)
            scores.append(guesses)

    return scores


trainMatrix = createMatrix(practiceTrainSentences)
testMatrix = createMatrix(practiceTestSentences)
actualTestMatrix = createMatrix(actualTestDf)


whichKMers = 1  #Max 4 because the smallest peptides have only 4 residues

trainLists = diffKMers(trainMatrix,actualTestMatrix,whichKMers)  #Tests from test data
#results = callClassifier(trainLists,trainingLabels)

#Tests from training samples
#trainLists = diffKMers(trainMatrix,testMatrix,whichKMers)
results = callClassifier(trainLists,practiceTrainLabels)

file_path = 'ans.txt'
sys.stdout = open(file_path, "w")
for x in results[0]:
    print(x)
#print("Results: ",results)
