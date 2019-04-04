import pandas as pd
import numpy as np
from collections import Counter
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix


df=pd.read_csv('guardian.csv')
print(df.sample(10))
print(Counter(df["category"]))

# preprocessing the data
def clean_str(string):
    string=re.sub(r"\n","",string)
    string=re.sub(r"\r","",string)
    string=re.sub(r"[0-9]","",string)
    string=re.sub(r"\"","",string)
    string=re.sub(r"\'","",string)
    return string.strip().lower()

X=[]
for i in range(df.shape[0]):
    X.append(clean_str(df.iloc[i][0]))

y=np.array(df["category"])

# splitting triining and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=5)


model=Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),
('clf',OneVsRestClassifier(LinearSVC(class_weight="balanced")))])


parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
               'tfidf__use_idf': (True, False)}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X, y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)


# Fitting the model with optimal parameters
model=Pipeline([('vectorizer',CountVectorizer(ngram_range=(1,1))),('tfidf',TfidfTransformer(use_idf=True)),
('clf',OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

# fittinh model with training data
model.fit(X_train,y_train)

# predicting on the whole dataset
training_data_pred=model.predict(X_train)
print('Training accuracy: %s'%accuracy_score(training_data_pred,y_train))

# predicting on the whole dataset
test_data_pred=model.predict(X_test)
print('Testing data accuracy: %s'%accuracy_score(test_data_pred,y_test))

# predicting on the whole dataset
data_pred=model.predict(X)
print('Overall data accuracy: %s'%accuracy_score(data_pred,y))
