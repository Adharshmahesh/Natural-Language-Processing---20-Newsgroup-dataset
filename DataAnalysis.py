import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter

#Pie chart for 20 newsgroups

from sklearn.datasets import fetch_20newsgroups
news_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'), shuffle=True, random_state=42)

fig = plt.figure(figsize=(7,7))
t, s = np.unique(news_train.target_names, return_counts=True)
patches, t = plt.pie(s, shadow=True)

plt.legend(patches, news_train.target_names, loc=(0.75,0.25))
plt.axis('equal')
plt.show()

fig1 = plt.figure(figsize = (7,3),facecolor='w', edgecolor='k')
t1, s1 = np.unique(news_train.target_names, return_counts=True)
plt.bar(t1,s1)
plt.xticks(rotation=90)
plt.title('Distribution of each class in 20 Newsgroups data set')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
#Zipf's Law plot

def zipfs(text,low=1,up=99):

    counter = Counter(' '.join(text).split())
    
    fig, ax = plt.subplots(1, 1)
    
     
    f = [fr for word, fr in counter.most_common()]
    low = np.percentile(f, low)
    up = np.percentile(f, up)
    f = [fr for fr in f if fr > low and fr <= up]
    
    r = range(1, len(f) + 1)
    
    ax.plot(r, f)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Zipfs law')
    
    plt.show()

    
    return ax


#Heaps law plot

def heaps(text):

    l = [len(t) for t in text]
    n = [len(set(t.split())) for t in text]
    
    
    fig, ax = plt.subplots(1, 1)
        
    ax.scatter(l, n)
    
    ax.set_xlabel('Length of document')
    ax.set_ylabel('No of unique tokens')
    ax.set_title('Heaps law')
    
    plt.show()

    return ax

zipfs(news_train.data)
heaps(news_train.data)

#suspicious indices
le = [len(t) for t in news_train.data]
u = [len(set(t.split())) for t in news_train.data]
suspicious_indice = [i for i, l in enumerate(le) if l >= 58000 and l <= 60000 and u[i] < 1500]
print(suspicious_indice)
