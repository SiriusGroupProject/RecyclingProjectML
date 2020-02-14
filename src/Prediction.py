import glob
import numpy as np
import pandas as pd
from matplotlib import image
import pickle

filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

data_dir = '..\\data\\*'
data = []
for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "\\*"):
        data.append(image.imread(image_dir))
# make a prediction
ynew = model.predict_classes(np.array(data))
# show the inputs and predicted outputs
for i in range(len(ynew)):
	print("Predicted=%s" % ( ynew[i]))