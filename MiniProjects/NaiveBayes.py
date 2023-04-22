#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import sys
import math
import collections


# In[2]:


from utility import GetListOfWordsAndNumberOffiles
from utility import ReadFile
from utility import stop_words


# In[3]:


if(len(sys.argv) == 3):
    training_path = sys.argv[1]
    testing_path = sys.argv[2]    
else:
    sys.exit("Please give right number of arguments-TRAINING & TESTING PATH containing both ham and spam folder>                                                     TEST PATH containing both ham and spam folder>")


# In[4]:


#We need to find 
#P(ham) = number of documents that belong to ham / Total Number of documents
#P(spam) = number of documents that belong to spam / Total Number of documents
#P(ham|bodyText) = (P(ham) * P(bodyText|ham)) / P(bodyText)
#P(bodyText|spam) = P(word1|spam) * P(word2|spam)*.....
#P(bodyText|ham) = P(word1|ham) * P(word2|ham)*.....
#P(word1|spam) = count of words that belong to spam / Total count of words that belong to spam 
#P(word1|ham) = count of words that belong to ham / Total count of words that belong to ham    

#For new word that is not seen yet in test document
#P(new-word|ham) or P(new-word|spam) = 0
#This would make the product zero and so we can solve this by applying logs
# if (log(P(ham|bodyText)) > log(P(spam|bodyText)))
#    return 'ham'
#else:
#    return 'spam'
#    
#log(P(ham|bodyText)) = log(P(ham)) + log(P(bodyText|ham))
#                     = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .....
    
#P(word1|ham) = (count of words that belong to ham + 1)/
#              (total number of words that belong to ham + number of distinct words in training)
#P(word1|spam) = (count of words that belong to spam + 1)/
#                 (total number of words that belong to spam + number of distinct words in training)    
#Path of the folder for ham & spam for train and test

HamFolderPath = training_path + '/ham'
SpamFolderPath = training_path + '/spam'

#All words in ham and spam folders and their counts  
NumberOfHam, NumberOfSpam = 0, 0
ListOfWordsInham = []
ListOfWordsInspam = []
ListOfWordsInham,NumberOfHam = GetListOfWordsAndNumberOffiles(HamFolderPath)
ListOfWordsInspam,NumberOfSpam = GetListOfWordsAndNumberOffiles(SpamFolderPath)


# In[ ]:


#Function to Find P(ham) and P(spam), by calculating the number of ham/spam documents and total number of documents
def FindPHamOrSpam(HamOrSpam):
    if HamOrSpam == "spam":
        P_spam = NumberOfSpam/(NumberOfSpam + NumberOfHam)
        return P_spam
    else:
        P_ham = NumberOfHam/(NumberOfSpam + NumberOfHam)
        return P_ham


# In[ ]:


#Next we will find the distinct words and its count in both spam and ham
HamDictionary = dict(collections.Counter(w.lower() for w in ListOfWordsInham))
SpamDictionary = dict(collections.Counter(w.lower() for w in ListOfWordsInspam))

#making bag of words for both ham and spam and further counting each Distinct word in it
bagOfWords = ListOfWordsInham + ListOfWordsInspam
BagOfWordsDict = collections.Counter(bagOfWords)

def UpdateCountOfMissingWords(AllWords,HamSpamWords):
    for words in AllWords:
        if words not in HamSpamWords:
            HamSpamWords[words] = 0
            
#get missing words in each Ham and Spam list and adding them and intializing their count= 0
UpdateCountOfMissingWords(BagOfWordsDict,HamDictionary)
UpdateCountOfMissingWords(BagOfWordsDict,SpamDictionary)


# In[ ]:


#P(word1|ham) = (count of word1 belonging to category ham + 1)/
#              (total number of words belonging to ham + number of distinct words in training database)
#P(word1|spam) = (count of word1 belonging to category spam + 1)/
#                 (total number of words belonging to spam + number of distinct words in training database)  
#Here, Counter contains total number of words belonging to ham/spam plus number of distinct words 
#in training dataset as we updated all the missing words in dictionary too
ProbOf_HamWords = dict()
ProbOf_SpamWords = dict()
def FindProbOfWords(classifier,removeStopWords):
    Counter = 0
    if(removeStopWords ==1):
            for word in stop_words:
                if word in HamDictionary:
                    del HamDictionary[word]
                if word in SpamDictionary:
                    del SpamDictionary[word]
                if word in BagOfWordsDict:
                    del BagOfWordsDict[word]     
    if classifier == "ham":
        for word in HamDictionary:
            Counter += (HamDictionary[word] + 1)
        for word in HamDictionary:
            ProbOf_HamWords[word] = math.log((HamDictionary[word] + 1)/Counter ,2)
    elif classifier == "spam":
        for word in SpamDictionary:
            Counter += (SpamDictionary[word] + 1)
        for word in SpamDictionary:
            ProbOf_SpamWords[word] = math.log((SpamDictionary[word] + 1)/Counter ,2) 

#caluculating probability for each word in ham and Spam folders 
FindProbOfWords("ham",0)
FindProbOfWords("spam",0) 


# In[ ]:


#Finally classify the emails as ham or spam    
def PredictHamOrSpam(pathToFile, classifier):
    ProbOfHam = 0 
    ProbOfSpam = 0 
    InCorrectlyClassified = 0
    NumberOfFiles = 0
                   
    if classifier == "spam":
        for fileName in os.listdir(pathToFile):
            words =ReadFile(fileName,pathToFile)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ProbOfHam = math.log(FindPHamOrSpam("ham"),2)
            ProbOfSpam = math.log(FindPHamOrSpam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .... 
            for word in words:
                if word in ProbOf_HamWords:
                    ProbOfHam += ProbOf_HamWords[word]
                if word in ProbOf_SpamWords:
                    ProbOfSpam += ProbOf_SpamWords[word]
            NumberOfFiles +=1
            if(ProbOfHam >= ProbOfSpam):
                InCorrectlyClassified+=1
    if classifier == "ham":
        for fileName in os.listdir(pathToFile):
            words =ReadFile(fileName,pathToFile)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ProbOfHam = math.log(FindPHamOrSpam("ham"),2)
            ProbOfSpam = math.log(FindPHamOrSpam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + ....            
            for word in words:
                if word in ProbOf_HamWords:
                    ProbOfHam += ProbOf_HamWords[word]
                if word in ProbOf_SpamWords:
                    ProbOfSpam += ProbOf_SpamWords[word]
            NumberOfFiles +=1
            if(ProbOfHam <= ProbOfSpam):
                InCorrectlyClassified+=1
    return InCorrectlyClassified,NumberOfFiles 


# In[ ]:


print("Naive Bayes with stop words for Ham & Spam test emails :")  

HamTest_Path = testing_path + '\ham'
SpamTest_Path = testing_path + '\spam'        
IncorrectlyClassifiedHam,TotalHamEmails = PredictHamOrSpam(HamTest_Path, "ham")
IncorrectlyClassifiedSpam,TotalSpamEmails = PredictHamOrSpam(SpamTest_Path,"spam")
HamClassificationAccuracy = round(((TotalHamEmails - IncorrectlyClassifiedHam )/(TotalHamEmails ))*100,2)
SpamClassificationAccuracy = round(((TotalSpamEmails -  IncorrectlyClassifiedSpam )/(TotalSpamEmails))*100,2)
AllEmailClassified = TotalHamEmails + TotalSpamEmails
TotalIncorrectClassified = IncorrectlyClassifiedHam + IncorrectlyClassifiedSpam
OverAllAccuracy = round(((AllEmailClassified  - TotalIncorrectClassified )/AllEmailClassified)*100,2)


# In[ ]:


print("\nTotal number of files: ", AllEmailClassified)
print("\nCalculating Accuracy over Ham Emails")
print("Total number of Ham Emails: ", TotalHamEmails)
print("Number of Emails Classified as Ham: ", TotalHamEmails - IncorrectlyClassifiedHam)
print("Number of Emails Classified as Spam: ",IncorrectlyClassifiedHam)
print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(HamClassificationAccuracy) + "%")

print("\nCalculating Accuracy over Spam Emails")
print("Total number of Spam Emails: ", TotalSpamEmails)
print("Number of Emails Classified as Spam: ", TotalSpamEmails - IncorrectlyClassifiedSpam)
print("Number of Emails Classified as Ham: ",IncorrectlyClassifiedSpam)
print("\nNaive Bayes Accuracy For Spam Emails Classification: " + str(SpamClassificationAccuracy) + "%") 

print("\nNaive Bayes Total accuracy for Test Emails: " + str(OverAllAccuracy) + "%")

print("\n")


# In[ ]:


print("Naive Bayes after removing stop words")
FindProbOfWords("ham",1)
FindProbOfWords("spam",1) 

IncorrectlyClassifiedHam,TotalHamEmails = PredictHamOrSpam(HamTest_Path, "ham")
IncorrectlyClassifiedSpam,TotalSpamEmails = PredictHamOrSpam(SpamTest_Path,"spam")
HamClassificationAccuracy = round(((TotalHamEmails - IncorrectlyClassifiedHam )/(TotalHamEmails ))*100,2)
SpamClassificationAccuracy = round(((TotalSpamEmails -  IncorrectlyClassifiedSpam )/(TotalSpamEmails))*100,2)
AllEmailClassified = TotalHamEmails + TotalSpamEmails
TotalIncorrectClassified = IncorrectlyClassifiedHam + IncorrectlyClassifiedSpam
OverAllAccuracy = round(((AllEmailClassified  - TotalIncorrectClassified )/AllEmailClassified)*100,2)


# In[ ]:


print("\nTotal number of files: ", AllEmailClassified)
print("\nCalculating Accuracy over Ham Emails")
print("Total number of Ham Emails: ", TotalHamEmails)
print("Number of Emails Classified as Ham: ", TotalHamEmails - IncorrectlyClassifiedHam)
print("Number of Emails Classified as Spam: ",IncorrectlyClassifiedHam)
print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(HamClassificationAccuracy) + "%")

print("\nCalculating Accuracy over Spam Emails")
print("Total number of Spam Emails: ", TotalSpamEmails)
print("Number of Emails Classified as Spam: ", TotalSpamEmails - IncorrectlyClassifiedSpam)
print("Number of Emails Classified as Ham: ",IncorrectlyClassifiedSpam)
print("\nNaive Bayes Accuracy For Spam Emails Classification: " + str(SpamClassificationAccuracy) + "%") 

print("\nNaive Bayes Total accuracy for Test Emails: " + str(OverAllAccuracy) + "%")

print("\n")


# In[ ]:




