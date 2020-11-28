#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT CELL
import pandas as pd
import string
import re
from collections import Counter
from scipy.spatial import distance
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
# from jupyterthemes import jtplot
from string import digits
import glob
import errno 
import math


# In[2]:


#Part 1


# In[3]:


trainpos = []
trainneg = []
testpos = []
testneg = []

stop_words = []
with open('./Dataset/stop_words.txt',encoding='ISO-8859-1') as f:
    stop_words = [line.rstrip() for line in f]
    


# In[4]:


paths = ['./Dataset/train/pos/*.txt', './Dataset/train/neg/*.txt', './Dataset/test/pos/*.txt', './Dataset/test/neg/*.txt']
sent = []
labels = ['pos','neg','pos','neg']

for path in paths:

    files = glob.glob(path) 
    l = []
    for name in files: 
        try: 
            
            with open(name,encoding='ISO-8859-1') as f: 
                p = f.read()
               
                l.append(p)
            
        except IOError as exc: 
            print(exc)
            if exc.errno != errno.EISDIR: 
                raise 
    sent.append(l)


# In[5]:


def rmpunc(para):
    return para.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))


# In[6]:


#Cleaning

for i, sen in enumerate(sent):
    Revs = pd.DataFrame(sen, columns =['Reviews'])
    
    #To Lowercase
    Revs['Reviews'] = Revs['Reviews'].str.lower()

    #Removing stopwords
    Revs['Reviews'] = Revs['Reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #Removing Punctuation
    Revs['Reviews'] = Revs.apply(lambda row: rmpunc(row["Reviews"]),axis=1 )
    
    sent[i] = Revs['Reviews'].tolist()


# In[7]:


trainpos = sent[0]
poses = ['pos'] * len(trainpos)

trainneg = sent[1]
negs = ['neg'] * len(trainneg)


trainrevs = trainpos + trainneg
Train_Y = poses + negs


testpos = sent[2]
poses = ['pos'] * len(testpos)

testneg = sent[3]
negs = ['neg'] * len(testneg)


testrevs = testpos + testneg
Test_Y = poses + negs


# In[8]:


#Making Vocab
all_text = ' '.join(trainrevs + testrevs)

vocab = list(set(all_text.split()))
# display(len(vocab))
# display(vocab)


# In[9]:


dictTrain = {
    'pos': Counter((' '.join(trainpos)).split()),
    'neg': Counter((' '.join(trainneg)).split())
    
}

dictTest = {
    'pos': Counter((' '.join(testpos)).split()),
    'neg': Counter((' '.join(testneg)).split())
    
}




# In[10]:


totaltrainrevs = len(trainrevs)
classes = ['pos', 'neg']
totalclassreviews = {
    'pos': len(trainpos), 'neg':len(trainneg)
}




# In[11]:


def train_naive_bayes(dictTrain, totaltrainrevs, totalclassreviews, vocab):
    logprior = {}
    loglikelihood = {}
    for c in classes:
        logprior[c] = math.log(totalclassreviews[c] / totaltrainrevs)
        loglikelihood[c] = {}
   
    for word in vocab:
        for c in dictTrain: 
            loglikelihood[c][word] = math.log((dictTrain[c][word] + 1) / ( sum(dictTrain[c].values()) + len(vocab)) )
    
    
    

    return logprior, loglikelihood


# In[12]:


logprior, loglikelihood = train_naive_bayes(dictTrain, totaltrainrevs, totalclassreviews, vocab)


# In[13]:


def hmap(Review, vocab, logprior, loglikelihood, classes):
    rev_words = Review.split()
    Sum = {}


    for c in classes:
        Sum[c] = logprior[c]
        result = {k: loglikelihood[c][k] for k in (loglikelihood[c].keys() & rev_words)}
        Sum[c]+= sum(result.values())
            
    argmax = max(Sum, key=Sum.get) 
    
    return argmax
    


def test_naive_bayes(testrevs, logprior, loglikelihood, vocab, classes):
    testrevs = np.array(testrevs) 

    
    predicted_labels = []
    for Review in np.nditer(testrevs):
        
        predicted_labels.append(hmap(str(Review), vocab, logprior,loglikelihood, classes))

    
    
    return predicted_labels

def match(col1, col2):
    if col1 == col2:
        return 1
    else:
        return 0

def get_accuracy(predicted_labels, actual_labels):
    prediction = pd.DataFrame({'Predicted Sentiment':predicted_labels, 'Actual Sentiment': actual_labels })
    
 
    
    prediction["Count"] = prediction.apply(lambda x: match(x['Predicted Sentiment'], x['Actual Sentiment']), axis=1)


    matches = prediction["Count"].sum()
    samples = prediction.shape[0]
    
    accuracy = matches / samples
    return accuracy


# In[14]:


predicted_labels = test_naive_bayes(testrevs,logprior, loglikelihood, vocab, classes)


# In[15]:


#Accuracy
accuracy = get_accuracy(predicted_labels,Test_Y)
display(accuracy)


# In[16]:


#Part 2


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

vectorizer = CountVectorizer(vocabulary=vocab)
Train_X = (vectorizer.fit_transform(trainrevs))
Test_X = (vectorizer.fit_transform(testrevs))




clf = MultinomialNB()
clf.fit(Train_X, Train_Y)


# In[18]:


predicted_labels = clf.predict(Test_X)
actual_labels = Test_Y
accuracy = accuracy_score(actual_labels, predicted_labels)
cm = confusion_matrix(actual_labels, predicted_labels)


# In[19]:


# Confusion matrix and accuracy
display(accuracy)
display(cm)


# In[ ]:




