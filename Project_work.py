# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:09:19 2024

@author: Vindhya Boddupalli
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#heart = pd.read_csv("C:/Users/HP/OneDrive/Documents/Kaggle_Dataset/Heart_Disease_Prediction_Final.csv")
heart = pd.read_csv("https://github.com/bvindhya26/heart_attack_prediction/blob/master/Heart_Disease_Prediction_Final.csv")
heart.shape
print("File size is :", heart.shape)
heart.head()
heart.tail()
heart.describe()
print(heart.describe())
heart.info()

heart.isnull().sum()

heart.groupby('Sex')['index'].nunique().plot.bar()

heart.groupby('Chest pain type')['index'].nunique().plot.bar()

heart.groupby('Heart Disease')['index'].nunique().plot.bar()

X = heart.iloc[:,1:14]

print(X)

y = heart['Heart Disease']

print(y)


fig = plt.figure(figsize=(12, 12), dpi=150, facecolor='#fafafa')
gs = fig.add_gridspec(4, 3)
gs.update(wspace=0.1, hspace=0.4)

background_color = "#fafafa"

plot = 0
for row in range(0, 1):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

import seaborn as sns



sns.kdeplot(heart['Age'] )
plt.show()
sns.kdeplot(heart['Sex'] )     
plt.show()

sns.kdeplot(heart['Exercise angina'] )     
plt.show()


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from imblearn.over_sampling import SMOTE

over_sample = SMOTE()
X_train_os, y_train_os = over_sample.fit_resample(X_train, y_train.ravel())


from sklearn.linear_model import LogisticRegression
logModel=LogisticRegression(solver='lbfgs', max_iter=10000)
logModel.fit(X_train_os, y_train_os)
predictions = logModel.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test,predictions)*100
print(classification_report(y_test, predictions))
print('Logistic Regression accuracy score: {0:0.4f}'.format(accuracy_score(y_test,predictions)*100))

import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train_os, y_train_os)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)*100))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train_os, y_train_os)
y_predict = clf.predict(X_test)

from sklearn import metrics
print(classification_report(y_test, y_predict))
print('Random Forest Classifier accuracy score: {0:0.04f}'.format(metrics.accuracy_score(y_test, y_predict)*100))

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100,max_depth=6,min_samples_split=2,min_weight_fraction_leaf =0.0,n_jobs=-1)
clf.fit(X_train_os, y_train_os)
#print(clf.score(X_test, y_test)*100)
y_predx = clf.predict(X_test)
print(classification_report(y_test, y_predx))
print('ExtraTreesClassifier accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predx)*100))

from sklearn.svm import SVC  
clf = SVC(kernel='linear') 
  
#fitting
clf.fit(X_train_os, y_train_os) 
y_predsvc=clf.predict(X_test)

print('SVM accuracy score: {0:0.4f}'.format(accuracy_score(y_test,y_predsvc)*100))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=250, p=2, weights = 'uniform')

knn.fit(X_train_os, y_train_os)
y_predknn = knn.predict(X_test)

print('KNeighborsClassifier accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predknn)*100))

plt.figure(figsize = (20,10))
sns.heatmap(heart.corr(), cmap = 'crest', annot = True)
plt.show()

from tensorflow.keras.models import Sequential #Helps to create Forward and backward propogation
from tensorflow.keras.layers import Dense, Dropout #Helps to create neurons in ANN
#from tensorflow.keras.layers import ReLU #activation functions

classifier = Sequential()

classifier.add(Dense(units = 12, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 7, activation = 'relu'))
classifier.add(Dropout(0.75))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

classifier.summary()

import tensorflow as tf
early_stopping=tf.keras.callbacks.EarlyStopping(
    min_delta=0.0001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

history=classifier.fit(X_train_os,y_train_os,validation_split=0.3,batch_size=20,epochs=500,callbacks=early_stopping)

history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:, ['loss']], "#6daa9f", label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']],"#774571", label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")

plt.show()

history_df = pd.DataFrame(history.history)

plt.plot(history_df.loc[:, ['accuracy']], "#6daa9f", label='Training accuracy')
plt.plot(history_df.loc[:, ['val_accuracy']], "#774571", label='Validation accuracy')

plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_predct = classifier.predict(X_test)
y_predct = (y_predct > 0.5)
print(classification_report(y_test, y_predct))
print('ANN accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_predct)*100))
