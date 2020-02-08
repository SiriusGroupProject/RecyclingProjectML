
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import keras.backend as K
import struct;print(struct.calcsize("P") * 8)
import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

img_width, img_height = 480, 320
data_dir = '..\\data\\*'

image_data = []
label_data = []

for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "\\*"):
        image = Image.open(image_dir)
        image_data.append(image)
        label_data.append(sub_directory)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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

X_train, X_test, y_train, y_test = train_test_split(
    image_data, label_data, test_size=0.25, random_state=42)

for i in X_train:
    print(i)

#for image, label in zip(image_data, label_data):