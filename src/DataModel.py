from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.backend as K
import glob
import numpy as np
import pandas as pd
from matplotlib import image
import pickle
from matplotlib import pyplot

from sklearn.model_selection import train_test_split

img_width, img_height = 480, 320
data_dir = '..\\data\\*'

image_data = []
label_data = []

for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "\\*"):
        data = image.imread(image_dir)
        # display the array of pixels as an image
        '''
        pyplot.imshow(data)
        pyplot.show()
        '''
        image_data.append(data)
        label_data.append(sub_directory)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

le = preprocessing.LabelEncoder()
s = pd.Series(label_data)

levels = pd.factorize(s)

print(np.array(levels[0]))
model.fit(np.array(image_data), np.array(levels[0]), validation_split=0.25, epochs=5, batch_size=10)
# save the model to disk
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
