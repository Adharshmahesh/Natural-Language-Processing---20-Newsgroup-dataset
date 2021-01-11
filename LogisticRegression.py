import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter


from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', remove=['headers','footers','quotes'])


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=True, preprocessor=None, tokenizer=None, stop_words='english', ngram_range=(1, 2), analyzer='word', max_df=1.0, min_df=1, max_features=None, binary=True)
X_train_counts = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

twenty_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)

pipeline = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('lbfgs', LogisticRegression())])

parameter = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search = GridSearchCV(pipeline, parameter,cv = 2)
grid_search.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search.best_score_))

print("Best estimated parameters:", grid_search.best_estimator_)


clf = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('lbfgs', LogisticRegression(C=1.0, class_weight=None, dual=False,fit_intercept=True, intercept_scaling=1,l1_ratio=None, max_iter=100,multi_class='auto', n_jobs=None,penalty='l2', random_state=None,solver='lbfgs', tol=0.0001, verbose=0,warm_start=False))])

clf.fit(twenty_train.data, twenty_train.target)

predicted = clf.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted == twenty_test.target))