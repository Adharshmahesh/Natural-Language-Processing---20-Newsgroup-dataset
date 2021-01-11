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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

twenty_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)

from sklearn.model_selection import GridSearchCV
pipeline = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', DecisionTreeClassifier())])

parameter = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search = GridSearchCV(pipeline, parameter,cv = 5)
grid_search.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search.best_score_))

print("Best estimated parameters:", grid_search.best_estimator_)

clf = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        presort='deprecated', random_state=None,
                                        splitter='best'))],
         verbose=False)

clf.fit(twenty_train.data, twenty_train.target)

predicted = clf.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted == twenty_test.target))