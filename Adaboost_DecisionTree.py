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

twenty_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)


x_train, x_val, y_train, y_val = train_test_split(X_train_tf, twenty_train.target, test_size=0.2,stratify=twenty_train.target)

parameter = {'random_state' :[None]}

param = {'learning_rate' : [0.01,0.1,1]}

grid_search = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier()), param_grid = parameter, cv = 3)
grid_search.fit(x_train, y_train)


print("Best cross-validation 5-fold score: {:.2f}".format(grid_search.best_score_))

print("Best estimated parameters:", grid_search.best_estimator_)

test_clf1 = Pipeline([('vect', count_vector),('tfidf', TfidfTransformer()),('clf', AdaBoostClassifier(algorithm='SAMME.R',base_estimator=DecisionTreeClassifier(ccp_alpha=0.0,class_weight=None,criterion='gini',max_depth=None,max_features=None,max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,presort='deprecated',random_state=None,splitter='best'),learning_rate=1.0, n_estimators=50, random_state=None))])

test_clf1.fit(twenty_train.data, twenty_train.target)

predicted1 = test_clf1.predict(twenty_test.data)
print("Test accuracy:", np.mean(predicted1 == twenty_test.target))

