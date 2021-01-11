import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.datasets import fetch_20newsgroups
news_train = fetch_20newsgroups(subset='train', remove=['headers','footers','quotes'])
news_test = fetch_20newsgroups(subset='test', remove=['headers','footers','quotes'], shuffle=True, random_state=42)


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(lowercase=True, preprocessor=None, tokenizer=None, stop_words='english', ngram_range=(1, 2), analyzer='word', max_df=1.0, min_df=1, max_features=None, binary=True)
x_train_counts = count_vector.fit_transform(news_train.data)
#x_test_counts = count_vector.transform(news_test.data)



from sklearn.feature_extraction.text import TfidfTransformer
tf_trans = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_trans.transform(x_train_counts)
#x_test_tf = tf_trans.transform(x_test_counts)

# Chi square test
#chi2_selector = SelectKBest(chi2, 900000)
#x_train_tfchi = chi2_selector.fit_transform(x_train_tf, news_train.target) 
#X_test_tfidf_sel2 = chi2_selector.transform(x_test_tf)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from sklearn.model_selection import GridSearchCV

#Linear SVC
x_train, x_val, y_train, y_val = train_test_split(x_train_tf, news_train.target, test_size=0.2,stratify=news_train.target)

alpha = [0.01, 0.1, 0.5, 1, 1.5,2, 2.5, 3]

grid_search = GridSearchCV(LinearSVC(), param_grid=dict(C=alpha), cv=5)
grid_search.fit(x_train,y_train)

print("Best cross-validation 5-fold score:",grid_search.best_score_)

print("Best estimated parameters:", grid_search.best_estimator_)

grid_search = grid_search.best_estimator_
y_valpredict = grid_search.predict(x_val)

print(confusion_matrix(y_valpredict,y_val))
print(classification_report(y_valpredict,y_val))


test_clf = Pipeline([('vect', count_vector),('tfidf', TfidfTransformer()),('clf', LinearSVC(C=2, class_weight=None, dual=True, fit_intercept = True, intercept_scaling = 1, loss='squared_hinge', max_iter=1000, multi_class = 'ovr', penalty = 'l2', random_state = None, tol = 0.0001, verbose=0))])
test_clf.fit(news_train.data, news_train.target)

predicted = test_clf.predict(news_test.data)
print("Test accuracy:", np.mean(predicted == news_test.target))

# SGDC Classifier

alpha1 = [0.001, 0.01, 0.5, 0.1]

grid_search1 = GridSearchCV(SGDClassifier(), param_grid = dict(alpha = alpha1), cv=5)
grid_search1.fit(x_train,y_train)

print("Best cross-validation 5-fold score: {:.2f}".format(grid_search1.best_score_))

print("Best estimated parameters:", grid_search1.best_estimator_)

test_clf1 = Pipeline([('vect', count_vector),('tfidf', TfidfTransformer()),('clf', SGDClassifier(alpha=0.001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='hinge',
              max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='l2',
              power_t=0.5, random_state=None, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False))])
test_clf1.fit(news_train.data, news_train.target)

predicted1 = test_clf1.predict(news_test.data)
print("Test accuracy:", np.mean(predicted1 == news_test.target))





