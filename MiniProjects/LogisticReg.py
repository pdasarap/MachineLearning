#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import re
import collections
import codecs
import numpy

from utility import initiliazeMatrix
from utility import stop_words

# In[ ]:


if (len(sys.argv) != 6):  
    sys.exit("Please give only valid Arguments- \n<path to TRAIN FOLDER that has both ham and spam folder>\n<path to TEST FOLDER that has both ham and spam folder>\n<yes or no to remove stop words>\n<Regularization parameters>\n<iteration>")
else:
    training = sys.argv[1]
    testing = sys.argv[2]
    Stop = sys.argv[3]
    Lamda = float(sys.argv[4])
    Iteration = sys.argv[5]


# In[ ]:


ham_List = list()
spam_List = list()
count_TrainHam = 0
count_TrainSpam = 0
ProbHam_Dict = dict()
ProbSpam_Dict = dict()
learningRate = 0.001
regularization = Lamda



# In[ ]:


bias = 0
xnode = 1
directory_Ham = training + '/ham'
directory_Spam = training + '/spam'
test_Ham = testing + '/ham'
test_Spam = testing + '/spam'

# Regular expression to clean the data given in train data- ham and spam folders
regex = re.compile(r'[A-Za-z0-9\']')

def FileOpen(filename, path):
    fileHandler = codecs.open(path + "\\" + filename, 'rU','latin-1')
    # codecs handles -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1651: character maps to <undefined>
    words = [Findwords.lower() for Findwords in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
    fileHandler.close()
    return words


# In[ ]:


def browseDirectory(path):
    wordList = list()
    fileCount = 0
    for files in os.listdir(path):
        if files.endswith(".txt"):
            wordList += FileOpen(files, path)
            fileCount += 1
    return wordList, fileCount


# In[ ]:


# iterating through train to get the list of ham words used to form combined bag of words
ham_List, count_TrainHam = browseDirectory(directory_Ham)
spam_List, count_TrainSpam = browseDirectory(directory_Spam)

# iterating through test to get the list of ham words used to form combined bag of words
hamTest, count_TestHam = browseDirectory(test_Ham)
SpamTest, count_TestSpam = browseDirectory(test_Spam)

def removeStopWords():
    for word in stop_words:
        if word in ham_List:
            i = 0
            lengthh=len(ham_List)
            while (i < lengthh):
                if (ham_List[i] == word):
                    ham_List.remove(word)
                    lengthh = lengthh - 1
                    continue
                i = i + 1
        if word in spam_List:
            i = 0
            lengths=len(spam_List)
            while (i < lengths):
                if (spam_List[i] == word):
                    spam_List.remove(word)
                    lengths = lengths - 1
                    continue
                i = i + 1
        if word in hamTest:
            i = 0
            lengthht=len(hamTest)
            while (i < lengthht):
                if (hamTest[i] == word):
                    hamTest.remove(word)
                    lengthht = lengthht - 1
                    continue
                i = i + 1
        if word in SpamTest:
            i = 0
            lengthst=len(SpamTest)
            while (i < lengthst):
                if (SpamTest[i] == word):
                    SpamTest.remove(word)
                    lengthst = lengthst - 1
                    continue
                i = i + 1


if (Stop == "yes"):
    removeStopWords()


# In[ ]:


# collections.Counter counts the number of occurence of memebers in list
raw_Ham = dict(collections.Counter(w.lower() for w in ham_List))
dict_Ham = dict((k, int(v)) for k, v in raw_Ham.items())
raw_Spam = dict(collections.Counter(w.lower() for w in spam_List))
dict_Spam = dict((k, int(v)) for k, v in raw_Spam.items())

bagOfWords = ham_List + spam_List
dict_BagOfWords = collections.Counter(bagOfWords)
list_BagOfWords = list(dict_BagOfWords.keys())
Target_List = list()  # final value of ham or spam, ham = 1 & spam = 0
totalFiles = count_TrainHam + count_TrainSpam

raw_TestHam = dict(collections.Counter(w.lower() for w in hamTest))
dict_TestHam = dict((k, int(v)) for k, v in raw_TestHam.items())
raw_TestSpam = dict(collections.Counter(w.lower() for w in SpamTest))
dict_TestSpam = dict((k, int(v)) for k, v in raw_TestSpam.items())

# correct it to testham/spam
BagOfWords_test = hamTest + SpamTest
DictBagOfWords_test = collections.Counter(BagOfWords_test)
ListBagOfWords_test = list(DictBagOfWords_test.keys())
TargetList_test = list()  # final value of ham or spam, ham = 1 & spam = 0
totalTestFiles = count_TestHam + count_TestSpam


# In[ ]:


# initialize matrix to zero and use list comprehension to create this matrix

featureMatrixTrain = initiliazeMatrix(totalFiles, len(list_BagOfWords))
featureMatrixTest = initiliazeMatrix(totalTestFiles, len(ListBagOfWords_test))

rowMatrix = 0
testRowMatrix = 0

sigMoid_List = list()  # for each row
for i in range(totalFiles):
    sigMoid_List.append(-1)
    Target_List.append(-1)

for i in range(totalTestFiles):
    TargetList_test.append(-1)

weightOfFeature = list()

for feature in range(len(list_BagOfWords)):
    weightOfFeature.append(0)


# In[ ]:


def makeMatrix(featureMatrix, path, listBagOfWords, rowMatrix, classifier, TargetList):
    for fileName in os.listdir(path):
        words = FileOpen(fileName, path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in listBagOfWords:
                column = listBagOfWords.index(key)
                featureMatrix[rowMatrix][column] = temp[key]
        if (classifier == "ham"):
            TargetList[rowMatrix] = 0
        elif (classifier == "spam"):
            TargetList[rowMatrix] = 1
        rowMatrix += 1
    return featureMatrix, rowMatrix, TargetList


# In[ ]:


#train matrix including ham and spam
featureMatrixTrain, rowMatrix, Target_List = makeMatrix(featureMatrixTrain, directory_Ham, list_BagOfWords, rowMatrix,
                                                       "ham", Target_List)
featureMatrixTrain, rowMatrix, Target_List = makeMatrix(featureMatrixTrain, directory_Spam, list_BagOfWords, rowMatrix,
                                                       "spam", Target_List)

featureMatrixTest, testRowMatrix, TargetList_test = makeMatrix(featureMatrixTest, test_Ham, ListBagOfWords_test,
                                                              testRowMatrix, "ham", TargetList_test)
featureMatrixTest, testRowMatrix, TargetList_test = makeMatrix(featureMatrixTest, test_Spam, ListBagOfWords_test,
                                                              testRowMatrix, "spam", TargetList_test)


# In[ ]:


# for each column
def sigmoid(x):
    den = (1 + numpy.exp(-x))
    sigma = 1 / den
    return sigma


# In[ ]:


# Calculate for each file
def sigmoidFunction(totalFiles, totalFeatures, featureMatrix):
    global sigMoid_List
    for files in range(totalFiles):
        summation = 1.0

        for features in range(totalFeatures):
            summation += featureMatrix[files][features] * weightOfFeature[features]
        sigMoid_List[files] = sigmoid(summation)


# In[ ]:


def calculateWeightUpdate(totalFiles, numberOfFeature, featureMatrix, TargetList):
    global sigMoid_List

    for feature in range(numberOfFeature):
        weight = bias
        for files in range(totalFiles):
            freq = featureMatrix[files][feature]
            y = TargetList[files]
            sigmoid_Value = sigMoid_List[files]
            weight += freq * (y - sigmoid_Value)

        oldW = weightOfFeature[feature]
        weightOfFeature[feature] += ((weight * learningRate) - (learningRate * regularization * oldW))

    return weightOfFeature


# In[ ]:


def trainingFunction(totalFiles, numbeOffeatures, trainFeatureMatrix, TargetList):
    sigmoidFunction(totalFiles, numbeOffeatures, featureMatrixTrain)
    calculateWeightUpdate(totalFiles, numbeOffeatures, featureMatrixTrain, Target_List)


# In[ ]:


def classifyData():
    correct_Ham = 0
    incorrect_Ham = 0
    correct_Spam = 0
    incorrect_Spam = 0
    idx=0
    for file in range(totalTestFiles):
        print('TestFile : '+str(idx+1))
        summation = 1.0
        for i in range(len(ListBagOfWords_test)):
            word = ListBagOfWords_test[i]

            if word in list_BagOfWords:
                index = list_BagOfWords.index(word)
                weight = weightOfFeature[index]
                wordcount = featureMatrixTest[file][i]

                summation += weight * wordcount

        sigSum = sigmoid(summation)
        if (TargetList_test[file] == 0):
            if sigSum < 0.5:
                correct_Ham += 1.0
            else:
                incorrect_Ham += 1.0
        else:
            if sigSum >= 0.5:
                correct_Spam += 1.0
            else:
                incorrect_Spam += 1.0
        idx += 1
    print("Accuracy on Ham:" + str(round((correct_Ham / (correct_Ham + incorrect_Ham)) * 100,3)))
    print("Accuracy on Spam:" + str(round((correct_Spam / (correct_Spam + incorrect_Spam)) * 100,3)))
    print("Overall Accuracy :" + str(round(((correct_Ham+correct_Spam) / (correct_Ham + incorrect_Ham+correct_Spam + incorrect_Spam)) * 100,3)))


# In[ ]:


print("Training the algorithm - ")
for i in range(int(Iteration)):
    print(i, end=' ')
    trainingFunction(totalFiles, len(list_BagOfWords), featureMatrixTrain, Target_List)


print("Training completed successfully")
print("\nPlease wait while classifying the data..\nThis may take few minutes")
classifyData()


