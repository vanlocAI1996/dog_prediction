import os
import cv2
import numpy as np
import pandas as pd

root = 'data'
def read_image(root):
	folders = os.listdir(root)
	image_infos = {}
	means = []
	var_list = []
	stds = []
	labels = []
	for folder in folders:
		folder_path = os.path.join(root, folder)
		for image in os.listdir(folder_path):
			image_path = os.path.join(folder_path, image)
			image = cv2.imread(image_path)
			mean =  np.mean(image)
			var = np.var(image)
			std = np.std(image)	
			means.append(mean)
			var_list.append(var)
			stds.append(std)
			if folder == 'positive':
				labels.append(1)
			else:
				labels.append(0)

	image_infos = {'mean': means, 'std': stds, 'var': var_list, 'label': labels}
	return image_infos

def save_to_csv(image_infos):
	df = pd.DataFrame(image_infos, columns=['mean', 'std', 'var', 'label'])
	df.to_csv('data/data.csv', index=None, header=True)
	print('File saved')
