import numpy as np 
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml 

n_estimators_yaml = yaml.safe_load(open("params.yaml",'r'))['model_building']['n_estimators']
learning_rate_yaml = yaml.safe_load(open("params.yaml",'r'))['model_building']['learning_rate']

train_data = pd.read_csv("./data/feature_eng/train_bow.csv") 

X_train = train_data.iloc[:,:-1].values 
y_train = train_data.iloc[:,-1].values

clf = GradientBoostingClassifier(n_estimators=n_estimators_yaml, learning_rate=learning_rate_yaml)

clf.fit(X_train,y_train) 

# save the model
pickle.dump(clf, open("model.pkl", "wb"))


