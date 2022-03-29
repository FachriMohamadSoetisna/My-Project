#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:24:41 2021

@author: Fachri Mohamad Soetisna & Yohannes Irfon
"""
# Importing necessary library
import re
import numpy as np
import timeit
import os
import math
import matplotlib.pyplot as plt
from nltk import sent_tokenize

start = timeit.default_timer()

# Importing files
text_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
text = []
for File in text_files:
    text.append(open(File,errors='ignore').read())



# Create Lists for sentences
a_list = []
for a in range(len(text)):
    a_list.append(sent_tokenize(text[a]))
    

# Method for create wordlist
wordList = {}
def generateWordList(sentenceList):
    tokenizedSentence = []
    for currentSentence in sentenceList:
        currentSentence = re.sub('[^A-Za-z ]','' , currentSentence)
        currentTokenized = currentSentence.lower().split()
        for currentWord in currentTokenized:
            wordList[currentWord] = 0
        tokenizedSentence.append(currentSentence.lower().split())
    return tokenizedSentence, wordList


# Method for find TF
def findTF(tokenizedList, wordList):
    featureList = wordList.keys()
    for currentTokenList in tokenizedList:
        currentTF = wordList.copy()
        for currentWord in currentTokenList:
            currentTF[currentWord] = currentTF[currentWord] + 1
            
        currentVector = []
        for currentKey in featureList:
            currentVector.append(currentTF[currentKey])
        vectors.append(currentVector)
    return vectors, featureList

# Generate wordlist
kumpulan_tokenizedList = []
for b in range(len(a_list)):
   for idx in range(len(a_list[b])):
       currentList = a_list[b]
       tokenizedList, wordList = generateWordList(currentList)
   kumpulan_tokenizedList.append(tokenizedList)
   tokenizedList = []
   currentList = []
   idx = 0

# Create vectors
vectors = []
for c in range(len(kumpulan_tokenizedList)):
   currentTokenizedList = kumpulan_tokenizedList[c]
   vectors, idxtoWord = findTF(currentTokenizedList, wordList)

# Method for search text
def findTF2(tokenizedList, wordList):
    kumpulan_vector = []
    featureList = wordList.keys()
    for currentTokenList in tokenizedList:
        currentTF = wordList.copy()
        for currentWord in currentTokenList:
            currentTF[currentWord] = currentTF[currentWord] + 1
            
        currentVector = []
        for currentKey in featureList:
            currentVector.append(currentTF[currentKey])
        kumpulan_vector.append(currentVector)
    return kumpulan_vector, featureList

# Create Search for text
text_search = []
for d in range(len(text_files)):
    currentTokenizedList = kumpulan_tokenizedList[d]
    kumpulan_vector, idxtoWord = findTF2(currentTokenizedList, wordList)
    text_search.append(kumpulan_vector)

vectors_copy = vectors
text_doc = []
for x in range(len(text_files)):
    for z in range(len(text_search[x])):
        text_doc.append(text_files[x])
        
text_doc_index = []
for m in range(len(text_doc)):
    text_doc_index.append(m)

# Normalizing the vector
normalized_vectors = []
for e in range(len(vectors)):
    normalized_vectors.append(vectors[e] / np.linalg.norm(vectors[e])) 

normalized_copy = normalized_vectors

# Pairing the vectors with cosine distance
def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
        
dic = []
scores = []
for i in range(len(normalized_vectors)):
    for j in range(len(normalized_copy)):
        cosine = cosine_similarity(normalized_vectors[i],normalized_copy[j])
        scores.append(cosine)
        dic.append([i,j])
        cosine = 0

# Sort scores & outlier analisys
score_copy = scores

sortedScores = sorted(score_copy)

sorted_array = np.array(sortedScores)

# Sorted scores visualization
plt.hist(sorted_array, alpha=0.5)

# Make List for outliers
score_contek = []
for z in range(len(scores)):
    if scores[z] != 1 and scores[z] >= 0.8 :
        score_contek.append(scores[z])
# Make list for sentence pair
pasangan_nyontek = []
for f in range(len(score_contek)):
    for g in range(len(scores)): 
        if score_contek[f] == scores[g]:
            pasangan_nyontek.append(dic[g])
            
pasangan_kalimat = []
for h in range(len(pasangan_nyontek)):
    if h%2 == 0 :
        pasangan_kalimat.append(pasangan_nyontek[h])
        
temp_list = list(zip(pasangan_kalimat, score_contek))

# Make list for suspected documents name
pasangan_dokumen_1 = []
pasangan_dokumen_2 = []
for k in range(len(pasangan_kalimat)):
    currentPasangan = pasangan_kalimat[k]
    for l in range(len(currentPasangan)):
        for n in range(len(text_doc_index)):
            if currentPasangan[l] == text_doc_index[n] and l == 0:
                pasangan_dokumen_1.append(text_doc[n])
            elif currentPasangan[l] == text_doc_index[n] and l == 1:
                pasangan_dokumen_2.append(text_doc[n])
                
pasangan_doc_dict = {}
for o in range(len(pasangan_dokumen_1)):
    pasangan_doc_dict[o] = [pasangan_dokumen_1[o],pasangan_dokumen_2[o]]
    


doc_score = list(zip(pasangan_doc_dict.values(), score_contek))

print(doc_score)

