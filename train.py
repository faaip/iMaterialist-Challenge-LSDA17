import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
import argparse
from keras import applications
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from keras.utils.np_utils import to_categorical

img_width, img_height = 224, 224  # image dimensions

# cli arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lr", "--learning-rate", required=True,
                help="Learning rate for Adam", type=float)
ap.add_argument('--retrain-all', dest='feature',
                    action='store_false')

args = ap.parse_args()
learning_rate = args.learning_rate
retrain_bottle = args.retrain_all
print(retrain_bottle)

# paths
top_model_path = 'models/entire_model.h5'
features_path = 'features/'
train_data_dir = '../data/train/'
validation_data_dir = '../data/valid/'

# hyper parameters
epochs = 82
batch_size = 16

model = applications.VGG16(include_top=False, weights='imagenet')

tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                          write_graph=True, write_images=True)


def train_bottleneck():
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train, verbose=1)

    np.save(features_path + 'bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation, verbose=1)

    np.save(features_path + 'bottleneck_features_validation.npy',
            bottleneck_features_validation)


train_bottleneck()

# labels for training data
datagen_top = ImageDataGenerator(rescale=1. / 255)
generator_top = datagen_top.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

# load the bottleneck features saved earlier
train_data = np.load(features_path + 'bottleneck_features_train.npy')

# get the class lebels for the training data, in the original order
train_labels = generator_top.classes

# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)

# labels for validation features
generator_top = datagen_top.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_validation_samples = len(generator_top.filenames)

validation_data = np.load(features_path + 'bottleneck_features_validation.npy')

validation_labels = generator_top.classes
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

# TRAIN TOP MODEL
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

adam = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=adam,
              loss='categorical_crossentropy', metrics=['accuracy'])

# https://stackoverflow.com/questions/43388186/keras-why-my-val-acc-suddenly-drops-at-epoch-42-50
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                              patience=3, min_lr=0.00001)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                                           baseline=None)

# now augment the data to improve accuracy
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
)

model_info = model.fit_generator(datagen.flow(train_data, train_labels, batch_size=batch_size),
                                 samples_per_epoch=train_data.shape[0],
                                 epochs=epochs,
                                 validation_data=(validation_data, validation_labels), verbose=1,
                                 callbacks=[reduce_lr, tb_callback])
# Save model
model.save(top_model_path)

(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))
