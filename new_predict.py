from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import pandas as pd

BATCH_SIZE = 16
IMG_WIDTH, IMG_HEIGHT = 224, 224  
TEST_DATA_DIR = 'data/'

vgg16_model = applications.VGG16(include_top=True, weights='imagenet')

def predict():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            TEST_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            color_mode="rgb",
            shuffle = False,
            class_mode=None,
            batch_size=1)

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    # make prediction
    y_prob = vgg16_model.predict_generator(test_generator,steps = nb_samples,
                                           verbose=1)
    y_classes = y_prob.argmax(axis=-1)
    print(y_classes)

    # create dataframe
    predictions_df = pd.DataFrame (y_classes,columns = ['predicted'])
    predictions_df['id'] = predictions_df.index + 1
    submission_df = predictions_df[predictions_df.columns[::-1]]
    file_name = "submission_" + datetime.now().strftime('%Y-%m-%d_%H_%M_%S') +".csv"
    submission_df.to_csv(file_name, index=False, header=True)


predict()

