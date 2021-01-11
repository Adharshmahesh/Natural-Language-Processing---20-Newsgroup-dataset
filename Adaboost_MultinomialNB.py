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
count_vector = CountVectorizer(lowercase=True, preprocessor=None, tokenizer=None, stop_words='english', ngram_range=(1, 2), analyzer='word', max_df=1.0, min_df=1, max_features=None, binary=True)
X_train_counts = count_vector.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

twenty_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)

alpha = [0.01, 0.1, 0.5, 1, 1.5]


parameter = {'random_state' :[None]}
#param = {'learning_rate' : [0.01,0.1,0.5,1]}
param = {'learning_rate' : [0.01,0.1,1]}
param1 = {'tf_id__smooth_idf' : (True, False),'tf_id__sublinear_tf' : (True, False)}
pipeline = Pipeline([('tf_id', TfidfVectorizer(stop_words = "english")), ('clf', AdaBoostClassifier(base_estimator = MultinomialNB()))])
grid_search = GridSearchCV(pipeline, param_grid = param1, cv = 3)

grid_search.fit(twenty_train.data, twenty_train.target)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search.best_score_))

print("Best estimated parameters:", grid_search.best_estimator_)

clf = Pipeline([('tfidf', TfidfVectorizer(stop_words = "english")),('clf', AdaBoostClassifier(algorithm='SAMME.R',
                                    base_estimator=MultinomialNB(alpha=1.0,
                                                                 class_prior=None,
                                                                 fit_prior=True),
                                    learning_rate=1.0, n_estimators=50,
                                    random_state=None))],
         verbose=False)

clf.fit(twenty_train.data, twenty_train.target)

predicted = clf.predict(twenty_test.data)

print("Test accuracy:", np.mean(predicted == twenty_test.target))