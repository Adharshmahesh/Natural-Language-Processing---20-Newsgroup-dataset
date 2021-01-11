import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import fetch_20newsgroups
news_train = fetch_20newsgroups(subset='train', remove=['headers','footers','quotes'])

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(lowercase=True, preprocessor=None, tokenizer=None, stop_words='english', ngram_range=(1, 2), analyzer='word', max_df=1.0, min_df=1, max_features=None, binary=True)
x_train_counts = count_vector.fit_transform(news_train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tf_trans = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_trans.transform(x_train_counts)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


news_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)

from sklearn.model_selection import GridSearchCV



from sklearn.multiclass import OneVsRestClassifier

pipeline = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', OneVsRestClassifier(LinearSVC(C=0.6)))])

parameter = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search = GridSearchCV(pipeline, parameter,cv = 5)
grid_search.fit(news_train.data, news_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search.best_score_))

print("Best estimated parameters:", grid_search.best_estimator_)


clf = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', OneVsRestClassifier(estimator=LinearSVC(C=0.6,
                                                         class_weight=None,
                                                         dual=True,
                                                         fit_intercept=True,
                                                         intercept_scaling=1,
                                                         loss='squared_hinge',
                                                         max_iter=1000,
                                                         multi_class='ovr',
                                                         penalty='l2',
                                                         random_state=None,
                                                         tol=0.0001,
                                                         verbose=0),
                                     n_jobs=None))],
         verbose=False)
clf.fit(news_train.data, news_train.target)

predicted = clf.predict(news_test.data)
print("Test accuracy:", np.mean(predicted == news_test.target))