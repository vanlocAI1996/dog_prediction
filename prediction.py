import pickle
import cv2
import numpy as np

filename = 'model/finalized_model.sav'
def load_model(filename):
	clf = pickle.load(open(filename, 'rb'))
	return clf

def create_test_data(image):
	image = cv2.imread(image)
	mean = np.mean(image)
	var = np.var(image)
	std = np.std(image)	
	test = np.array([[mean, var, std]])
	return test
test = create_test_data('data/5.jpg')
clf = load_model(filename)
predict = clf.predict(test)
if predict[0] == 0:
	print('Has no dog!')
else:
	print('Has dog!')