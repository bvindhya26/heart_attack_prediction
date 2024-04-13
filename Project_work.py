# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:09:19 2024

@author: Vindhya Boddupalli
"""

import numpy as np
import pandas as pd

heart = pd.read_csv("C:/Users/HP/OneDrive/Documents/Kaggle_Dataset/Heart_Disease_Prediction.csv")
heart.shape
print("File size is :", heart.shape)
heart.head()
heart.tail()
heart.describe()
heart.info()

heart.isnull().sum()

heart.groupby('Heart_Disease')['index'].nunique().plot.bar()

X = heart.iloc[:,1:14]

print(X)

y = heart['Heart_Disease']

print(y)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logModel=LogisticRegression(solver='lbfgs', max_iter=10000)
logModel.fit(X_train, y_train)
predictions = logModel.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)*100
print('Logistic Regression accuracy score: {0:0.4f}'.format(accuracy_score(y_test,predictions)*100))

import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)*100))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

from sklearn import metrics
print('Random Forest Classifier accuracy score: {0:0.04f}'.format(metrics.accuracy_score(y_test, y_predict)*100))

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100,max_depth=6,min_samples_split=2,min_weight_fraction_leaf =0.0,n_jobs=-1)
clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test)*100)
y_predx = clf.predict(X_test)
print('ExtraTreesClassifier accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predx)*100))


from tensorflow.keras.models import Sequential #Helps to create Forward and backward propogation
from tensorflow.keras.layers import Dense #Helps to create neurons in ANN
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU #activation functions

classifier = Sequential()

classifier.add(Dense(units = 12, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dense(units = 7, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])\

import tensorflow as tf
early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=800,callbacks=early_stopping)

y_predct = classifier.predict(X_test)
y_predct = (y_predct > 0.5)

print('ANN accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predct)*100))

from sklearn.svm import SVC  
clf = SVC(kernel='linear') 
  
#fitting x samples and y classes 
clf.fit(X_train, y_train) 
y_predsvc=clf.predict(X_test)

print('SVM accuracy score: {0:0.4f}'.format(accuracy_score(y_test,y_predsvc)*100))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)
y_predknn = knn.predict(X_test)

print('KNeighborsClassifier accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predknn)*100))

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

plt.figure(figsize = (20,10))
sns.heatmap(heart.corr(), cmap = 'crest', annot = True)
plt.show

fig = px.histogram(heart, x="chest_pain_type", title="Distribution of Chest Pain Types", barmode="group", color="chest_pain_type", width=600, height=400)
fig.show()