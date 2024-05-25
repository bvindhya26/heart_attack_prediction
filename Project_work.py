# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:09:19 2024

@author: Vindhya Boddupalli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

heart = pd.read_csv("C:/Users/HP/OneDrive/Documents/Kaggle_Dataset/Heart_Disease_Prediction_Final.csv")
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

heart.groupby('FBS over 120')['index'].nunique().plot.bar()

heart.groupby('Exercise angina')['index'].nunique().plot.bar()

heart.groupby('Number of vessels fluro')['index'].nunique().plot.bar()


X = heart.iloc[:,1:14]

print(X)

y = heart['Heart Disease']

print(y)

heart.groupby('Heart Disease')['index'].nunique().plot.bar()

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

heart.insert(15,'Heart Disease Binary',y)

print(heart)

plt.figure(figsize = (20,10))
sns.heatmap(heart.corr(), cmap = 'crest', annot = True)
plt.show()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.3, random_state=42)

print('Shape for training data', X_train.shape, y_train.shape)
print('Shape for testing data', X_test.shape, y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

X_train_scaled, X_test_scaled

from imblearn.over_sampling import SMOTE

over_sample = SMOTE()
X_train_os, y_train_os = over_sample.fit_resample(X_train, y_train.ravel())



from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

# using Softmax Regression (multi-class classification problem)
log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
# 'C' is hyprparameter for regularizing L2
# 'lbfgs' is Byoden-Fletcher-Goldfarb-Shanno(BFGS) algorithm
log_clf.fit(X_train_scaled, y_train)

# Let us predict all instances of training dataset X_train_scaled using the above trained model
y_train_predict = log_clf.predict(X_train_scaled)

log_accuracy = accuracy_score(y_train, y_train_predict)
log_precision = precision_score(y_train, y_train_predict, average='weighted')
log_recall = recall_score(y_train, y_train_predict, average='weighted')
log_f1_score = f1_score(y_train, y_train_predict, average='weighted')


print("Logistic Accuracy: ", log_accuracy)
print("Logistic Precision: ", log_precision)
print("Logistic Recall: ", log_recall)
print("Logistic F1 Score: ", log_f1_score)

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)

# Scaling is not needed for Decision Tree algorithm and hence for Random Forest and XGBoost algorithms as they 
# are also based on Decision Trees. Hence, not using scaled training dataset here

rnd_clf.fit(X_train, y_train)

# Let us predict all instances of training dataset X_train using the above trained model
y_train_predict = rnd_clf.predict(X_train)

rnd_accuracy = accuracy_score(y_train, y_train_predict)
rnd_precision = precision_score(y_train, y_train_predict, average='weighted')
rnd_recall = recall_score(y_train, y_train_predict, average='weighted')
rnd_f1_score = f1_score(y_train, y_train_predict, average='weighted')


print("Random Forest Accuracy: ", rnd_accuracy)
print("Random Forest Precision: ", rnd_precision)
print("Random Forest Recall: ", rnd_recall)
print("Random Forest F1 Score: ", rnd_f1_score)

import lightgbm

lgb_clf = lightgbm.LGBMClassifier()

lgb_clf.fit(X_train_scaled, y_train)

# Let us predict all instances of training dataset X_train using the above trained model
y_train_predict = lgb_clf.predict(X_train)

lgb_accuracy = accuracy_score(y_train, y_train_predict)
lgb_precision = precision_score(y_train, y_train_predict, average='weighted')
lgb_recall = recall_score(y_train, y_train_predict, average='weighted')
lgb_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("LightGBM Accuracy: ", lgb_accuracy)
print("LightGBM Precision: ", lgb_precision)
print("LightGBM Recall: ", lgb_recall)
print("LightGBM F1 Score: ", lgb_f1_score)

from sklearn.svm import SVC  
svm_clf = SVC(kernel='linear') 
  
#fitting
svm_clf.fit(X_train_scaled, y_train) 

# Let us predict all instances of training dataset X_train using the above trained model
y_train_predict = svm_clf.predict(X_train)

svm_accuracy = accuracy_score(y_train, y_train_predict)
svm_precision = precision_score(y_train, y_train_predict, average='weighted')
svm_recall = recall_score(y_train, y_train_predict, average='weighted')
svm_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("SVM Accuracy: ", lgb_accuracy)
print("SVM Precision: ", lgb_precision)
print("SVM Recall: ", lgb_recall)
print("SVM F1 Score: ", lgb_f1_score)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=250, p=2, weights = 'uniform')

#fitting
knn_clf.fit(X_train_scaled, y_train) 

# Let us predict all instances of training dataset X_train using the above trained model
y_train_predict = knn_clf.predict(X_train)

knn_accuracy = accuracy_score(y_train, y_train_predict)
knn_precision = precision_score(y_train, y_train_predict, average='weighted')
knn_recall = recall_score(y_train, y_train_predict, average='weighted')
knn_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("KNeighbors Accuracy: ", knn_accuracy)
print("KNeighbors Precision: ", knn_precision)
print("KNeighbors Recall: ", knn_recall)
print("KNeighbors F1 Score: ", knn_f1_score)


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

# function to calculate mean and standard deviation of each score (e.g. accuracy, precision, etc.)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42) 

log_cv_scores = cross_val_score(log_clf, X_train_scaled, y_train, cv=3, scoring="accuracy") 
display_scores(log_cv_scores)
log_cv_accuracy = log_cv_scores.mean()

y_train_pred = cross_val_predict(log_clf, X_train_scaled, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)
log_cv_precision = precision_score(y_train, y_train_pred, average='weighted')
log_cv_recall = recall_score(y_train, y_train_pred, average='weighted')
log_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')

print("Logistic CV Accuracy: ", log_cv_accuracy)
print("Logistic CV Precision: ", log_cv_precision)
print("Logistic CV Recall: ", log_cv_recall)
print("Logistic CV F1 Score: ", log_cv_f1_score)


rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=10 , random_state=42)
rnd_cv_scores = cross_val_score(rnd_clf, X_train, y_train, cv=3, scoring="accuracy") 
display_scores(rnd_cv_scores)
rnd_cv_accuracy = rnd_cv_scores.mean()

y_train_pred = cross_val_predict(rnd_clf, X_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)
rnd_cv_precision = precision_score(y_train, y_train_pred, average='weighted')
rnd_cv_recall = recall_score(y_train, y_train_pred, average='weighted')
rnd_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')

print("Random Forest CV Accuracy: ", rnd_cv_accuracy)
print("Random Forest CV Precision: ", rnd_cv_precision)
print("Random Forest CV Recall: ", rnd_cv_recall)
print("Random Forest CV F1 Score: ", rnd_cv_f1_score)

print("X_train_scaled shape:",X_train_scaled.shape)
print("y_train shape:", y_train.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test shape:", y_test.shape)


X_train_scaled_split = X_train_scaled[:301]
y_train_split = y_train[:301]
X_valid = X_train_scaled[301:]
y_valid = y_train[301:]

print("X_train_scaled_split shape:",X_train_scaled_split.shape)
print("y_train_split shape:", y_train_split.shape)
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)


X_mean = X_train_scaled_split.mean(axis=0, keepdims=True)
X_std = X_train_scaled_split.std(axis=0, keepdims=True)
X_train_scaled_split = (X_train_scaled_split - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test_scaled = (X_test_scaled - X_mean) / X_std
X_train_scaled_split = X_train_scaled_split[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test_scaled = X_test_scaled[..., np.newaxis]

print ("Shape of features", X_train_scaled_split.shape, X_valid.shape, X_test_scaled.shape)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

model = keras.Sequential()
n_features = 1

# First GRU layer
model.add(layers.GRU(units=100, return_sequences=True, input_shape=(1,n_features), activation='tanh'))
model.add(layers.Dropout(0.2))

# Second GRU layer
model.add(layers.GRU(units=150, return_sequences=True, input_shape=(1,n_features), activation='tanh'))
model.add(layers.Dropout(0.2))

# Third GRU layer
model.add(layers.GRU(units=100, activation='tanh'))
model.add(layers.Dropout(0.2))

# The output layer
model.add(layers.Dense(units=1, kernel_initializer='he_uniform', activation='linear'))

#model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.0005) , metrics = ['mean_squared_error'])

print(model.summary())

history = model.fit(X_train_scaled_split,y_train_split,epochs=100,batch_size=120, verbose=1, validation_data = (X_valid,y_valid))

import math

def model_score(model, X_train_scaled_split, y_train_split, X_valid, y_valid , X_test_scaled, y_test):
    print('Train Score:')
    train_score = model.evaluate(X_train_scaled_split, y_train_split, verbose=0)
    print("MSE: {:.5f} , RMSE: {:.2f}".format(train_score[0], math.sqrt(train_score[0])))

    print('Validation Score:')
    val_score = model.evaluate(X_valid, y_valid, verbose=0)
    print("MSE: {:.5f} , RMSE: {:.2f}".format (val_score[0], math.sqrt(val_score[0])))

    print('Test Score:')
    test_score = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("MSE: {:.5f} , RMSE: {:.2f}".format (test_score[0], math.sqrt(test_score[0])))


model_score(model, X_train_scaled_split, y_train_split ,X_valid, y_valid , X_test_scaled, y_test)

print(history.history.keys())
plt.plot(history.history['loss'])  # plotting train loss
plt.plot(history.history['val_loss'])  # plotting validation loss

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pred = model.predict(X_test_scaled)
print(pred)

pred = (pred>=0.5)
print(pred)
print(classification_report(y_test, pred))
print('ANN accuracy score: {0:0.4f}'.format(accuracy_score(y_test, pred)*100))
