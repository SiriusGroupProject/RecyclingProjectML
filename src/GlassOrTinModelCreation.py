from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.backend as K
import glob
import numpy as np
from matplotlib import image
import pickle
import os

img_width, img_height = 480, 320
data_dir = '../train_data/*'
image_data = []
label_data = []

for sub_directory in glob.glob(data_dir):
    for image_dir in glob.glob(sub_directory + "/*"):
        if os.path.basename(sub_directory).startswith("cam") or os.path.basename(sub_directory).startswith("teneke"):
            print(image_dir)
            data = image.imread(image_dir)
            image_data.append(data)
            label_data.append(sub_directory)


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)

le = preprocessing.LabelEncoder()
le.fit(label_data)
labels = le.transform(label_data)

input_test = labels.astype('float32')

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

print(input_test)
# Normalize data

history = model.fit(np.array(image_data), input_test, validation_split=0.20, epochs=5, batch_size=10)

# save the model to disk
filename = 'model_glass_vs_tin.sav'
pickle.dump(model, open(filename, 'wb'))
