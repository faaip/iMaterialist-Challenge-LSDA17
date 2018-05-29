import datetime
import json
import math
import time


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
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
MODEL_PATH = 'models/entire_model.h5'
TRAIN_DATA_DIR = '../data/train/'
VALIDATION_DATA_DIR = '../data/valid/'
SUBMISSION_DIR = 'submissions/'


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
# load model
model = load_model(MODEL_PATH)

generator_top = datagen_top.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False)

# Load json data
json_data = json.load(open('kaggle/test.json'))
test_dir = '../data/test/'

timestr = time.strftime("%Y%m%d-%H%M%S")

ids = []
predicted_labels = []
#file_name = SUBMISSION_DIR + time.strftime("%Y%m%d-%H%M%S") + 'submission.csv'
file_name = SUBMISSION_DIR + 'submission.csv'
try:
    df = pd.read_csv('submission.csv')
    start_index = df['id'].max()
    print("Starting at:",start_index)
except FileNotFoundError:
    print("Starting new submission")
    start_index = 0

for i in tqdm(json_data['images'][start_index:]):
    try:
        pred = get_prediction(test_dir + str(i['image_id']) + '.jpg')
        ids.append(i['image_id'])
        predicted_labels.append(pred)
    except FileNotFoundError:
        print('FILE NOT FOUND FOR', str(i['image_id']))
        ids.append(i['image_id'])
        predicted_labels.append(np.random.randint(0, 128))

    if int(i['image_id']) % 100 == 0:
        my_submission = pd.DataFrame(
            {'id': ids, 'predicted': predicted_labels})
        my_submission.to_csv(file_name, index=False)

# # save final
my_submission = pd.DataFrame({'id': ids, 'predicted': predicted_labels})
my_submission.to_csv(file_name, index=False)
