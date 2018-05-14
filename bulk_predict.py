from keras.models import Sequential

FULL_TOP_MODEL_PATH = 'models/bottleneck_fc_model.h5'  # the top layer

# Setup model
model = Sequential()
model.load(FULL_TOP_MODEL_PATH)

#
predictions = model.predict_classes(X_test.values, verbose=0)
predictions_df = pd.DataFrame (predictions,columns = ['predicted'])
predictions_df['id'] = predictions_df.index + 1
submission_df = predictions_df[predictions_df.columns[::-1]]
submission_df.to_csv("submission.csv", index=False, header=True)
submission_df.head()
