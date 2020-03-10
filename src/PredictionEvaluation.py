import glob
import numpy as np
import pandas as pd
from matplotlib import image
import pickle
import os

filename = 'model_glass_vs_tin.sav'
model = pickle.load(open(filename, 'rb'))

img_width, img_height = 480, 320
data_dir = '../test_data/*'
image_data = []
label_data = []

for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "/*"):
        if os.path.basename(sub_directory).startswith("cam") or os.path.basename(sub_directory).startswith("teneke"):
            data = image.imread(image_dir)
            image_data.append(data)
            if os.path.basename(sub_directory).startswith("cam"):
                label_data.append(0)
            else:
                label_data.append(1)
image_X = np.array(image_data);
ynew = model.predict(image_X)
score = model.evaluate(image_X, label_data, verbose=0)
print("Score of Glass or Tin Model")
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

'''
Model Score of Glass and Plastic
'''
filename = 'model_plastic_vs_glass.sav'
model = pickle.load(open(filename, 'rb'))
image_data = []
label_data = []

for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "/*"):
        if os.path.basename(sub_directory).startswith("cam") or os.path.basename(sub_directory).startswith("plastik"):
            data = image.imread(image_dir)
            image_data.append(data)
            if os.path.basename(sub_directory).startswith("cam"):
                label_data.append(0)
            else:
                label_data.append(1)


image_X = np.array(image_data);
ynew = model.predict(image_X)
score = model.evaluate(image_X, label_data, verbose=0)
print("Score of Glass or Plastic Model")
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

'''
Model Score of Glass and Plastic
'''
filename = 'model_plastic_vs_tin.sav'
model = pickle.load(open(filename, 'rb'))
image_data = []
label_data = []

for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "/*"):
        if os.path.basename(sub_directory).startswith("plastik") or os.path.basename(sub_directory).startswith("teneke"):
            data = image.imread(image_dir)
            image_data.append(data)
            if os.path.basename(sub_directory).startswith("plastik"):
                label_data.append(0)
            else:
                label_data.append(1)


image_X = np.array(image_data);
ynew = model.predict(image_X)
score = model.evaluate(image_X, label_data, verbose=0)
print("Score of Plastic or Tin Model")
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))