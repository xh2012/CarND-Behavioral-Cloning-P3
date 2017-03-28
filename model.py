# %load model_gen.py
# generator models for p3

import csv
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_dir = r"./traindata/four/"


samples = []
with open(os.path.join(data_dir, "driving_log.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    samples.pop(0)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

nb_train = len(train_samples) * 6
nb_valid = len(validation_samples) * 6

print("nb_train  = %d" % nb_train)
print("nb_valid  = %d" % nb_valid)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for sample in batch_samples:
                center, left, right, steering, throttle, brake, speed = sample

                image_center = cv2.imread(os.path.join(data_dir, center.strip()))
                image_left = cv2.imread(os.path.join(data_dir, left.strip()))
                image_right = cv2.imread(os.path.join(data_dir, right.strip()))

                correction = 0.2
                angle_center = float(steering)
                angle_left = angle_center + correction
                angle_right = angle_center - correction

                images.extend([image_center, image_left, image_right])
                angles.extend([angle_center, angle_left, angle_right])

                images_flipped = [np.fliplr(image_center), np.fliplr(image_left), np.fliplr(image_right)]
                angles_flipped = [- angle_center, -angle_left, -angle_right]

                images.extend(images_flipped)
                angles.extend(angles_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


# hyperparamenet batch size for training and validating
train_batch_size = 32
valid_batch_size = 32

train_generator = generator(train_samples, batch_size=train_batch_size)
validation_generator = generator(validation_samples, batch_size=valid_batch_size)


steps_per_epoch = nb_train / train_batch_size
validation_steps = nb_valid / valid_batch_size

print("train_batches  = %d" % steps_per_epoch)
print("valid_batches  = %d" % validation_steps)


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D,Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.backend import tensorflow_backend as ktf
from keras.layers.normalization import BatchNormalization


model = Sequential(
    [
        BatchNormalization(epsilon=1e-3, input_shape=(160,320,3)),
        Cropping2D(cropping=((50, 20), (0, 0))),
        Conv2D(24,5,5,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(36,5,5,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(48,5,5,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(64,3,3,activation="relu", border_mode="valid", subsample=(2,2)),
        Conv2D(64,3,3,activation="relu", border_mode="valid", subsample=(1,1)),
        Flatten(),
        Dropout(0.2),
        # Dense(1164),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(1)
    ]
)


model.summary()

plot_model(model, to_file="model.png",show_shapes=True)

model.compile(loss="mse", optimizer="adam")


his = model.fit_generator(generator = train_generator,
                          validation_data=validation_generator,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          epochs=3)

model.save("model.h5")


import matplotlib.pyplot as plt
print(his.history.keys())
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()
