import datetime
import json
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import applications
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

# predictor
vgg16_model = applications.VGG16(include_top=False, weights='imagenet')

# constants
BATCH_SIZE = 16
IMG_WIDTH, IMG_HEIGHT = 224, 224  # image dimensions
TOP_MODEL_WEIGHTS_PATH = 'models/bottleneck_fc_model.h5'  # the top layer
TRAIN_DATA_DIR = '../data/train/'
VALIDATION_DATA_DIR = '../data/valid/'
TEST_DIR = '../data/test/'


def get_prediction(image_path):
    #print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255
    image = np.expand_dims(image, axis=0)
    bottleneck_prediction = vgg16_model.predict(image)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    inID = class_predicted[0]

    class_dictionary = generator_top.class_indices

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]
    return label


print('Running main')
# Set variables for model
datagen_top = ImageDataGenerator(rescale=1. / 255)
generator_top = datagen_top.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

# load the bottleneck features saved earlier
train_data = np.load('bottleneck_features_train.npy')

# get the class lebels for the training data, in the original order
train_labels = generator_top.classes

# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)

# labels for validation features
generator_top = datagen_top.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False)

nb_validation_samples = len(generator_top.filenames)

validation_data = np.load('bottleneck_features_validation.npy')
validation_labels = generator_top.classes
validation_labels = to_categorical(
    validation_labels, num_classes=num_classes)

image = load_img(TEST_DIR +'/1.jpg', target_size=(224, 224))
image = img_to_array(image)

# important! otherwise the predictions will be '0'
image = image / 255
image = np.expand_dims(image, axis=0)
bottleneck_prediction = vgg16_model.predict(image)

# build top model
model = Sequential()
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.load_weights(TOP_MODEL_WEIGHTS_PATH)

# Load json data
json_data = json.load(open('kaggle/test.json'))

ids = []
predicted_labels = []
file_name = 'submission_' + str(datetime.datetime.now()) + '.csv'

for i in tqdm(json_data['images']):
    try:
        pred = get_prediction(test_dir + str(i['image_id']) + '.jpg')
        ids.append(i['image_id'])
        predicted_labels.append(pred)
    except FileNotFoundError:
        print('FILE NOT FOUND FOR', str(i['image_id']))
        ids.append(i['image_id'])
        predicted_labels.append(np.random.randint(0, 128))

    if int(i['image_id']) % 10 == 0:
        my_submission = pd.DataFrame(
            {'id': ids, 'predicted': predicted_labels})
        my_submission.to_csv(file_name, index=False)

# save final
my_submission = pd.DataFrame({'id': ids, 'predicted': predicted_labels})
my_submission.to_csv(file_name, index=False)
