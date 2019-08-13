import pandas as pd
import os
from sklearn import tree
import pickle

clf = tree.DecisionTreeClassifier()
root = 'data'
def read_csv_file(root):
	csv_path = os.path.join(root, 'data.csv')
	df = pd.read_csv(csv_path)
	return df.sample(frac=1)

def extract_training_data(df):
	labels = df['label']
	features = df.drop(['label'], axis=1)
	return features, labels

def train(features, labels):
	features = features.values
	labels = labels.values
	clf.fit(features, labels)
	return clf

df = read_csv_file(root)
features, labels = extract_training_data(df)
clf = train(features, labels)
filename = 'model/finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))