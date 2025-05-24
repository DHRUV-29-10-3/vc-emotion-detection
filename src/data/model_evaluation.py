import numpy as np 
import pandas as pd 
import pickle  
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 
import json

clf = pickle.load(open("model.pkl", "rb")) 

test_data = pd.read_csv("./data/feature_eng/test_bow.csv")
X_test = test_data.iloc[:,:-1].values 
y_test = test_data.iloc[:,-1].values

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metric_dict = {
    "Acc" : accuracy, 
    "Pre" : precision, 
    "Re" : recall, 
    "AUC" : auc
}
with open("metrics.json", "w") as file:
    json.dump(metric_dict, file, indent = 4)