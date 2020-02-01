
from __future__ import print_function
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import nltk
import os
import __future__
import re 
import numpy
import csv
import pandas as pd
import numpy as np
import math

numberOfDocuments = 404
postprocessingCount = 0

def readTranscripts():
    dir = "C:\\Users\\cressm\\Desktop\\TCSS 554\\Assignment1\\Assignment1\\transcripts\\"
    words = []
    documents = []
    documentId = 1
    for filename in os.listdir(dir):
        with open(dir+filename, "r") as f:
            for line in f:
                for w in line.split():
                    words.append(w)
                    documents.append(documentId)
        documentId = documentId + 1
    d = {'word': words, 'document': documents}
    df = pd.DataFrame(data=d, columns=['word', 'document'])
    return df

def getStopWords():
    stopwordsFile = "C:\\Users\\cressm\\Desktop\\TCSS 554\\Assignment1\\Assignment1\\stopwords.txt"
    stopwords = []
    with open(stopwordsFile, "r") as f:
       for line in f:
         stopwords.extend(line.split())
    return stopwords

def removeStopWords(words, stopWords):
    words = words.drop(words[words['word'].isin(stopWords)].index)
    return words

def removeSpecialCharacterWords(words):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','""',' ','']
    words = words.drop(words[words['word'].isin(symbols)].index)
    return words

def stripWords(words):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','"',' ','']
    for index, row in words.iterrows():       
        newWord = str.strip(row['word'])
        newWord = str.strip(newWord, "`~!@#$%^&*()_-+={[}}|\\:;\"<,>.?/ ")
        newWord = re.sub('[^A-Za-z0-9]+', '', newWord)
        newWord = newWord.lower()
        words.iat[index,0] = newWord
    return words

def stemmWords(words):
    stemmer = PorterStemmer()
    for index, row in words.iterrows():
        if index < len(words.index):
            stemmedWord = stemmer.stem(row['word'])
            words.iat[index,0] = stemmedWord
    return words

def getWordFrequencyPerDocument(words):
    df = words[['word']]
    df = df.groupby('word').size().reset_index(name='tf')
    df = df.sort_values(by='tf', ascending=False)
    return df

def getSingleOccurenceWordCount(wordCounts):
    gk = wordCounts.groupby('word', as_index=False)['tf'].sum()
    gk.rename(columns = {'tf':'df'}, inplace = True) 
    singleOccurenceTable = gk[gk['df'] == 1]
    return (len(singleOccurenceTable.index))

def getUniqueWordCount(words):
    t = words[['word']]
    gk = t.groupby('word', as_index=False)
    return gk.ngroups

def getTop30(words):
    gk = words.groupby('word', as_index=False)['tf'].sum()
    gk = gk.sort_values(by='tf', ascending=False)
    gk = gk.head(31)
    return gk;

def getTop30DocFrequencey(top30, words):
    list = [] 
    for index, row in top30.iterrows():
        docRows = words[words['word'] == row['word']]
        gk = docRows.groupby(['word','document']).count()
        list.append(gk.index.size)
    top30['df'] = list
    return top30;

def getIDF(top30):
    idfs = []
    for index, row in top30.iterrows():
        idfs.append(math.log10(numberOfDocuments/row['df']))
    df['idf'] = idfs
    return df

def calculateIdf(df):
    df['TF.IDF'] = df['tf'] * df['idf']
    return df

def calculateScaledTF(df):
    scaledTF = []
    for index, row in df.iterrows():
        scaledTF.append((1+math.log10(row['tf'])) * math.log10(numberOfDocuments/row['df']))
    df['scaledTF'] = scaledTF
    return df;

def calculateProbability(df, top30):
    probabilities = []
    for index, row in df.iterrows():
        probabilities.append(row['tf']/postprocessingCount)
    df['probability'] = probabilities
    return df;


words = readTranscripts()
stopWords = getStopWords()

#Total Tokens
preprocessingCount = len(words)
print('Total Tokens: ' + str(preprocessingCount))

words = stripWords(words)

#Remove Garabage Special Character Tokens
words = removeSpecialCharacterWords(words);
print('Word Count After Garbage Clean Up: ' + str(len(words)))

#Remove Stop Words
words = removeStopWords(words, stopWords);
words = stemmWords(words)
postprocessingCount = len(words)
print('Word Count After Stop Word Removal: ' + str(len(words)))

#Unique Word Count
uniqueWordCount = getUniqueWordCount(words)
print('Unique Word Count: ' + str(uniqueWordCount))

#Average Words Per Doc
averageWordsPerDocument = postprocessingCount /numberOfDocuments
print('Average Word Count Per Document: ' + str(averageWordsPerDocument))

wordCount = getWordFrequencyPerDocument(words);

#Single Appearence Word Count
singleWordCount = getSingleOccurenceWordCount(wordCount)
print('Single Occurence Words: ' + str(singleWordCount))


top30 = getTop30(wordCount);
df = getTop30DocFrequencey(top30, words)
df = getIDF(df)
df = calculateIdf(df)
df = calculateScaledTF(df)
df = calculateProbability(df, top30)
export_csv = df.to_csv(r'C:\Users\cressm\Desktop\TCSS 554\Assignment1\Assignment1\export_dataframe.csv', index = None, header=True)






         