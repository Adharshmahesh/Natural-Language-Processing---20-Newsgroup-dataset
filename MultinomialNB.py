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
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

twenty_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)

pipeline = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', MultinomialNB())])

parameter = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search = GridSearchCV(pipeline, parameter,cv = 5)
grid_search.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search.best_score_))

print("Best estimated parameters:", grid_search.best_estimator_)

clf = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],
         verbose=False)

clf.fit(twenty_train.data, twenty_train.target)

predicted = clf.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted == twenty_test.target))

pipeline1 = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', ComplementNB())])
grid_search1 = GridSearchCV(pipeline1, parameter,cv = 5)
grid_search1.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search1.best_score_))

print("Best estimated parameters:", grid_search1.best_estimator_)

clf1 = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', ComplementNB(alpha=1.0, class_prior=None, fit_prior=True,
                              norm=False))],
         verbose=False)

clf1.fit(twenty_train.data, twenty_train.target)

predicted1 = clf1.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted1 == twenty_test.target))


from sklearn.multiclass import OneVsRestClassifier

pipeline3 = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', OneVsRestClassifier(MultinomialNB(alpha=1)))])


grid_search3 = GridSearchCV(pipeline3, parameter,cv = 5)
grid_search3.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search3.best_score_))

print("Best estimated parameters:", grid_search3.best_estimator_)

clf3 = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', OneVsRestClassifier(estimator=MultinomialNB(alpha=1,
                                                             class_prior=None,
                                                             fit_prior=True),
                                     n_jobs=None))],
         verbose=False)

clf3.fit(twenty_train.data, twenty_train.target)

predicted3 = clf3.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted3 == twenty_test.target))



pipeline4 = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', OneVsRestClassifier(ComplementNB(alpha=1)))])


grid_search4 = GridSearchCV(pipeline4, parameter,cv = 5)
grid_search4.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search4.best_score_))

print("Best estimated parameters:", grid_search4.best_estimator_)

clf4 = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', OneVsRestClassifier(estimator=ComplementNB(alpha=1,
                                                             class_prior=None,
                                                             fit_prior=True),
                                     n_jobs=None))],
         verbose=False)

clf4.fit(twenty_train.data, twenty_train.target)

predicted4 = clf4.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted4 == twenty_test.target))







