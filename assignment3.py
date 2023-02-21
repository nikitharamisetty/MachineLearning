#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import numpy as np


# In[2]:


path_train = "/Users/nkitharamisetty/Desktop/assignment3/train"
path_test = "/Users/nkitharamisetty/Desktop/assignment3/test"


# In[3]:


total_size = 0
size_spam = 0
size_ham = 0
path = os.listdir(path_train)
spam_word_count={}
ham_word_count = {}
total_word_count = {}
for i in path:
    if i == '.DS_Store':
         continue
    y = os.listdir(path_train+"//" + i)

    if i=="ham":
        for j in y:
            total_size = total_size + 1
            size_ham = size_ham + 1
            f = path_train+"//"+ i + "//" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in ham_word_count and word.isalpha():
                    ham_word_count[word] = 1
                    total_word_count[word] = 1
                elif word.isalpha():
                    ham_word_count[word] =  ham_word_count[word]+1
                    total_word_count[word] = total_word_count[word]+1

    else:
        for j in y:
            total_size += 1
            size_spam += 1
            f = path_train+"//"+ i + "//" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in spam_word_count and word.isalpha():
                    spam_word_count[word] = 1
                    total_word_count[word] = 1
                elif word.isalpha():
                    spam_word_count[word] += 1
                    total_word_count[word] += 1
print("Total Word Count:",len(total_word_count))


# Naive Bayes

# In[4]:


totalwords_spam = sum(spam_word_count.values())
totalwords_ham = sum(ham_word_count.values())
length = len(total_word_count)
count_spam = 0
count_ham = 0
cst = 0
cht = 0
size_test = 0
for i in path:
    if i == '.DS_Store':
         continue
    y = os.listdir(path_test+"//"+ i)
    for j in y:
        test_sham = {}
        size_test = size_test + 1
        f = path_test+"//"+ i + "//" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in test_sham and word.isalpha():
                test_sham[word] = 1
            elif word.isalpha():
                test_sham[word] = test_sham[word] + 1
        prob_spam = math.log(size_spam/total_size)
        prob_ham = math.log(size_ham/total_size)
        for k in test_sham:
            if spam_word_count.get(k) != None:
                prob_spam = prob_spam + math.log((spam_word_count.get(k)+1)/((totalwords_spam)+(length)))
            else:
                prob_spam = prob_spam + math.log((1)/((totalwords_spam)+(length)))
            if ham_word_count.get(k) != None:
                prob_ham = prob_ham + math.log((ham_word_count.get(k)+1)/((totalwords_ham)+(length)))
            else:
                prob_ham = prob_ham + math.log((1)/((totalwords_ham)+(length)))

            if prob_spam > prob_ham:
                count_spam = count_spam + 1
                if i=="spam":
                    cst = cst + 1
            elif prob_ham > prob_spam:
                count_ham = count_ham + 1
                if i=="ham":
                    cht = cht + 1
print("Number of files:",total_size)
print("Number of spam files:",size_spam)
print("Number of ham files:",size_ham)
print("Accuracy",(cst+cht)/(count_spam+count_ham))


# Logistic Regression

# In[32]:


iterations = 5
lambd = 0.01
eta = 0.01
log_total_word_count = list(total_word_count.keys())
mat = np.zeros((total_size,len(log_total_word_count)+1))
z = 0
for i in path:
    if i == '.DS_Store':
         continue
    y = os.listdir(path_train+"//"+ i)
    for j in y:
        log_word_count = {}
        f = path_train+"//"+ i + "//" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in log_word_count and word.isalpha():
                log_word_count[word] = 1
            elif word.isalpha():
                log_word_count[word] = log_word_count[word]+ 1
        for k in log_word_count:

            mat[z][log_total_word_count.index(k)] = log_word_count[k]
        if i=="spam":
            mat[z][len(log_total_word_count)] = 1
        z = z + 1


# In[28]:


def prob(w,x):
    s = 0
    for i in range(len(x)):
        s = s + (w[i]*x[i])
    try:
        p = math.exp(w[0]+s)/(1 + math.exp(w[0]+s))
    except:
        p = 1
    return p


# In[29]:


w_new = np.ones(len(total_word_count)+1)
w = np.ones(len(total_word_count)+1)
probab = np.ones(mat.shape[0])
for k in range(iterations):
    w = w_new.copy()
    w_new = np.ones(len(total_word_count)+1)
    for l in range(mat.shape[0]):
        probab[l] = prob(w,mat[l])
    for i in range(len(w)):
        temp = 0
        for j in range(mat.shape[0]):
            temp = temp + mat[j][i]*((mat[j][mat.shape[1]-1])-probab[j])
        w_new[i] = w[i]+ (lambd * temp) - (lambd*eta*w[i])


# In[30]:


mat_test = np.zeros((size_test,len(log_total_word_count)+1))
z = 0
for i in path:
    if i == '.DS_Store':
         continue  
    y = os.listdir(path_test+"//"+ i)
    for j in y:
        log_word_count = {}
        f = path_test+"//"+ i + "//" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in log_word_count and word.isalpha():
                log_word_count[word] = 1
            elif word.isalpha():
                log_word_count[word] += 1
        for k in log_word_count:
            if k in log_total_word_count:
                mat_test[z][log_total_word_count.index(k)] = log_word_count[k]
        if i=="spam":
            mat_test[z][len(log_total_word_count)] = 1
        z = z + 1


# In[31]:


th = 0
ts = 0
tt = 0
for i in range(mat_test.shape[0]):
    s = 0
    for j in range(mat_test.shape[1]-1):
        s = s + (w_new[j]*mat_test[i][j])
    s = s + w[0]
    tt += 1
    if mat_test[i][len(log_total_word_count)]==1 and s>0:
        ts += 1
    elif mat_test[i][len(log_total_word_count)]==0 and s<0:
        th += 1
print("Number of Iterations:"+ str(iterations))
print("Accuracy:",(ts+th)/tt)


# In[34]:


stopWords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
             "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
             "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't",
             "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from",
             "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
             "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
             "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
             "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
             "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
             "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
             "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
             "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
             "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
             "you're", "you've", "your", "yours", "yourself", "yourselves"]


# Naive Bayes with stop words

# In[35]:


path = os.listdir(path_train)
spam_word_count={}
ham_word_count = {}
total_word_count = {}
for i in path:
    if i == '.DS_Store':
         continue  
    y = os.listdir(path_train+"//"+ i)
    if i=="spam":
        for j in y:
            f = path_train+"//"+ i + "//" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in stopWords:
                    if word not in spam_word_count and word.isalpha():
                        spam_word_count[word] = 1
                        total_word_count[word] = 1
                    elif word.isalpha():
                        spam_word_count[word] += 1
                        total_word_count[word] += 1
    else:
        for j in y:
            f = path_train+"//"+ i + "//" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in stopWords:
                    if word not in ham_word_count and word.isalpha():
                        ham_word_count[word] = 1
                        total_word_count[word] = 1
                    elif word.isalpha():
                        ham_word_count[word] += 1
                        total_word_count[word] += 1

print("Total Word Count:",len(total_word_count))


# In[36]:


totalwords_spam = sum(spam_word_count.values())
totalwords_ham = sum(ham_word_count.values())
length = len(total_word_count)
count_spam = 0
count_ham = 0
cst = 0
cht = 0
for i in path:
    if i == '.DS_Store':
         continue  
    y = os.listdir(path_test+"//"+ i)
    for j in y:
        test_sh = {}
        f = path_test+"//"+ i + "//" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in stopWords:
                if word not in test_sh and word.isalpha():
                    test_sh[word] = 1
                elif word.isalpha():
                    test_sh[word] += 1
        prob_spam = math.log(size_spam/total_size)
        prob_ham = math.log(size_ham/total_size)
    
        for k in test_sh:
            if spam_word_count.get(k) != None:
                prob_spam = prob_spam + math.log((spam_word_count.get(k)+1)/((totalwords_spam)+(length)))
            else:
                prob_spam = prob_spam + math.log((1)/((totalwords_spam)+(length)))
            if ham_word_count.get(k) != None:
                prob_ham = prob_ham + math.log((ham_word_count.get(k)+1)/((totalwords_ham)+(length)))
            else:
                prob_ham = prob_ham + math.log((1)/((totalwords_ham)+(length)))

            if prob_spam > prob_ham:
                count_spam = count_spam + 1
                if i=="spam":
                    cst = cst + 1
            elif prob_ham > prob_spam:
                count_ham = count_ham + 1
                if i=="ham":
                    cht = cht + 1
print("Number of files:",total_size)
print("Number of spam files:",size_spam)
print("Number of ham files:",size_ham)
print("Accuracy",(cst+cht)/(count_spam+count_ham))


#  Logistic Regression with stop words

# In[37]:


log_total_word_count = list(total_word_count.keys())
mat = np.zeros((total_size,len(log_total_word_count)+1))
z = 0
for i in path:
    if i == '.DS_Store':
         continue  
    y = os.listdir(path_train+"//"+ i)
    for j in y:
        log_word_count = {}
        f = path_train+"//"+ i + "//" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in stopWords:
                if word not in log_word_count and word.isalpha():
                    log_word_count[word] = 1
                elif word.isalpha():
                    log_word_count[word] += 1
        for k in log_word_count:
            mat[z][log_total_word_count.index(k)] = log_word_count[k]
        if i=="spam":
            mat[z][len(log_total_word_count)] = 1
        z = z + 1


# In[39]:


w_new = np.ones(len(total_word_count)+1)
w = np.ones(len(total_word_count)+1)
for k in range(iterations):
    w = w_new.copy()
    w_new = np.ones(len(total_word_count)+1)
    for l in range(mat.shape[0]):
        probab[l] = prob(w,mat[l])
    for i in range(len(w)):
        temp = 0
        for j in range(mat.shape[0]):
            temp = temp + mat[j][i]*((mat[j][mat.shape[1]-1])-probab[j])
        w_new[i] = w[i]+ (lambd * temp) - (lambd*eta*w[i])


# In[40]:


mat_test = np.zeros((size_test,len(log_total_word_count)+1))
z = 0
for i in path:
    if i == '.DS_Store':
         continue  
    y = os.listdir(path_test+"//"+ i)
    for j in y:
        log_word_count = {}
        f = path_test+"//"+ i + "//" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in stopWords:
                if word not in log_word_count and word.isalpha():
                    log_word_count[word] = 1
                elif word.isalpha():
                    log_word_count[word] += 1
        for k in log_word_count:
            if k in log_total_word_count:
                mat_test[z][log_total_word_count.index(k)] = log_word_count[k]
        if i=="spam":
            mat_test[z][len(log_total_word_count)] = 1
        z = z + 1


# In[43]:


th = 0
ts = 0
tt = 0

for i in range(mat_test.shape[0]):
    s = 0
    for j in range(mat_test.shape[1]-1):
        s = s + (w_new[j]*mat_test[i][j])
    s = s + w[0]
    tt += 1
    if mat_test[i][len(log_total_word_count)]==1 and s>0:
        ts += 1
    elif mat_test[i][len(log_total_word_count)]==0 and s<0:
        th += 1
print("Iterations:",iterations)
print("Accuracy:",(ts+th)/tt)


# In[ ]:




