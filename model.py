import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, MaxPooling2D


def plot_loss(history):
    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    plt.savefig("mse_loss.png")

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.15  # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path = './data/IMG/'
                img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
                img_left = cv2.imread(path + batch_sample[1].split('/')[-1])
                img_right = cv2.imread(path + batch_sample[2].split('/')[-1])

                # add images and angles to data set
                images.extend([img_center, img_left, img_right])
                images.extend([np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
                steering_angles.extend([steering_center, steering_left, steering_right])
                steering_angles.extend([-steering_center, -steering_left, -steering_right])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
#model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * 6,
                    nb_epoch=3,
                    verbose=2)

model.save('model.h5')
print('model saved.')

plot_loss(history)
