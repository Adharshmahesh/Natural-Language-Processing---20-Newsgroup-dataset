import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter



from sklearn.datasets import fetch_20newsgroups
news_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'), shuffle=True, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV


news_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)
from sklearn.ensemble import VotingClassifier

pipeline1 = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('lbfgs', LogisticRegression())])
parameter1 = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search1 = GridSearchCV(pipeline1, parameter1,cv = 3)
grid_search1.fit(news_train.data, news_train.target)
lr = grid_search1.best_estimator_
print("Best cross-validation 5-fold score: {:.2f}".format(grid_search1.best_score_))

print("Best estimated parameters:", grid_search1.best_estimator_)

pipeline2 = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', RandomForestClassifier())])

parameter2 = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search2 = GridSearchCV(pipeline2, parameter2,cv = 3)
grid_search2.fit(news_train.data, news_train.target)
rf = grid_search2.best_estimator_
print("Best cross-validation 5-fold score: {:.2f}".format(grid_search2.best_score_))

print("Best estimated parameters:", grid_search2.best_estimator_)

pipeline3 = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', DecisionTreeClassifier())])

parameter3 = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
grid_search3 = GridSearchCV(pipeline3, parameter3,cv = 3)
grid_search3.fit(news_train.data, news_train.target)
dt = grid_search3.best_estimator_
print("Best cross-validation 5-fold score: {:.2f}".format(grid_search3.best_score_))

print("Best estimated parameters:", grid_search3.best_estimator_)

estimate=[('Logisticregression', lr), ('RandomForest', rf),('DecisionTree',dt)]
ens = VotingClassifier(estimate, voting='hard')
ens.fit(news_train.data, news_train.target)
#ypred_val_en   = ensemble.predict(X_testing)
y_test_pred = ens.predict(news_test.data)

print(np.mean(y_test_pred == news_test.target))
print(confusion_matrix(y_test_pred,news_test.target))




