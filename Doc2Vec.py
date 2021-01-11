from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import fetch_20newsgroups
news_train = fetch_20newsgroups(subset='train', remove=['headers','footers','quotes'])

news_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)


def labels(c, type_l):
    
    l = []
    for i, v in enumerate(c):
        labels = type_l + '_' + str(i)
        l.append(TaggedDocument(v.split(), [labels]))
    return l

def vector(model, size_c, size_v, type_v):
   
    vectors = np.zeros((size_c, size_v))
    for i in range(0, size_c):
        prefix = type_v + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors



X_train, X_test, y_train, y_test = train_test_split(news_train.data, news_train.target, random_state=1, test_size=0.2)
X_train = labels(X_train, 'Train')
X_test = labels(X_test, 'Test')
datatotal = X_train + X_test

dbow = Doc2Vec(dm=0, vector_size=500, negative=5, min_count=1, alpha=0.05, min_alpha=0.05)
dbow.build_vocab([x for x in tqdm(datatotal)])

for i in range(10):
    dbow.train(utils.shuffle([x for x in tqdm(datatotal)]), total_examples=len(datatotal), epochs=1)
    dbow.alpha -= 0.002
    dbow.min_alpha = dbow.alpha




dbow_train = vector(dbow, len(X_train), 500, 'Train')
dbow_test = vector(dbow, len(X_test), 500, 'Test')

#logreg = LinearSVC(C=0.05)
lrw = LogisticRegression(C=1e3)
#lrw.fit(dbow_train, y_train)
lrw = lrw.fit(dbow_train, y_train)
y_pred = lrw.predict(dbow_test)

from sklearn.metrics import classification_report
print("Accuracy is:",accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

