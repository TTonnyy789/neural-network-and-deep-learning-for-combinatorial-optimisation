#%%#
### Step-1 ######################################################################
### Change the environment into xgboost(conda)


## Import the necessary libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw, dtw
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_predict
from nixtlats import NixtlaClient
from nixtlats.date_features import CountryHolidays
from pytorch_forecasting import TimeSeriesDataSet
from keras.models import Sequential, save_model, load_model, save_model
from keras.layers import Dense, LSTM, Embedding, Input
from statsmodels.tools.eval_measures import rmse, rmspe
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1, l1_l2
# from graph_encoding import *
# from data_preprocessing import *



# data_1 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/7CT_s5_v7-7766.json')
# data_2 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/7CT_s6_v6-9122.json')
# data_3 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/11CT_s5_v13-6106.json')
# data_4 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/raw/11CT_s12_v34-6729.json')


# ## Print the data form the json file 
# print("-------------------------------------------------")
# print("Data from the json file")
# print("-------------------------------------------------")
# print(data_1)
# print("------------------------------------------------- \n")
# print(data_2)
# print("------------------------------------------------- \n")
# print(data_3)
# print("------------------------------------------------- \n")
# print(data_4)


### ---------------------------------------------------------------------------


#%%#
### Step-2-feasible classification-node2vec ########################################################
### Feasible grpah classification - node2vec
df = pd.read_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v4_graph_vectors_node2vec_mean_one.csv')

## Randomly shuffle the data
df_new = df.sample(frac=1).reset_index(drop=True)


## Split the data into train and test
X = df_new.drop(['label'], axis=1)
y = df_new['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## Sequential model
model = Sequential()
model.add(Dense(128, input_dim=64, activation='tanh', kernel_regularizer=l1_l2(0.001))) ## LeakyReLU
model.add(Dense(128, activation='tanh')) ## tanh
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))  ## Single neuron for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


## Training the model
history = model.fit(X_train, y_train, epochs=200, batch_size=128) ## epoch:200 > 100

## Evaluate the model
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = np.mean(y_pred == y_test)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("-----------------------------\n")
print('Accuracy:', accuracy)
print("-----------------------------\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("-----------------------------\n")
print("Confusion Matrix:\n", cm)
print("-----------------------------\n")
print("ROC AUC Score:", roc_auc)



## Precision-Recall Curve and AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)


## Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR score = {pr_auc:}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

## Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC score = {roc_auc:}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


## Visualize the epoch and loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()





## Recall on 0 label: 0.67 - v4_mean_multi
## Recall on 0 label: 0.61 - v4_sum_multi
## Recall on 0 label: 0.75 - v4_mean_one, ROC AUC Score: 0.88
## Recall on 0 label: 0.71 - v4_sum_one, ROC AUC Score: 0.85




### ---------------------------------------------------------------------------





#%%#
### Step-3-1-structure2vec for GraphClassification #################################################
df = pd.read_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v4_graph_vectors_structure2vec_sum.csv')

## Randomly shuffle the data
df_new = df.sample(frac=1).reset_index(drop=True)


## Split the data into train and test
X = df_new.drop(['label'], axis=1)
y = df_new['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Sequential model
model = Sequential()
model.add(Dense(2048, input_dim=64, activation='tanh'))
model.add(Dense(1024, activation='tanh'))
model.add(Dense(1024, activation='tanh'))
model.add(Dense(512, activation='tanh'))
model.add(Dense(512, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))  # Single neuron for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


## Training the model
model.fit(X_train, y_train, epochs=100, batch_size=64)


## Evaluate the model
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

## Accuracy, F1 score, Precision, Recall, ROC AUC
accuracy = np.mean(y_pred == y_test)
print('Accuracy: %.2f' % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred))

## Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

## ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)

## Precision-Recall Curve and AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)


## Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


## Plot ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()






#%%#
### Step-3-2-transformer for GraphClassification #################################################

df = pd.read_csv('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/data/processed/v4_graph_vectors_transformer_sum.csv')

## Randomly shuffle the data
df_new = df.sample(frac=1).reset_index(drop=True)


## Split the data into train and test
X = df_new.drop(['label'], axis=1)
y = df_new['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Sequential model
## for the activation function, since the value of the vector is very negative and positive, we use the linear activation function, in mean aggregation approach
model = Sequential()
model.add(Dense(2048, input_dim=64, activation='linear'))
model.add(Dense(1024, activation='linear'))
model.add(Dense(1024, activation='linear'))
model.add(Dense(512, activation='linear'))
model.add(Dense(512, activation='linear'))
model.add(Dense(256, activation='linear'))
model.add(Dense(256, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(16, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='sigmoid'))  # Single neuron for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


## Training the model
model.fit(X_train, y_train, epochs=500, batch_size=64)

## Evaluate the model
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

## Accuracy, F1 score, Precision, Recall, ROC AUC
accuracy = np.mean(y_pred == y_test)
print('Accuracy: %.2f' % (accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred))

## Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

## ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)

## Precision-Recall Curve and AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)


## Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


## Plot ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()




# %%
