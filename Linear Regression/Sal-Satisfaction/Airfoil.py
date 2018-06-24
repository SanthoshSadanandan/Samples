import tensorflow as tf
import numpy as np
import pandas as pd
import fileops
CSV_COLUMN_NAMES = ['Frquency','Angle_of_Attack','Chord_Length','Free_stream_velocity','Displacement','Sound_pressure_level']

data = pd.read_csv('airfoil_self_noise.csv')
# Delete rows with unknowns
data = data.dropna()
# Shuffle the data - test
np.random.seed(None)

sound_pressure_level = data.pop('Sound_pressure_level')
feat_columns = data

# Create feature column and estimator

lin_reg = tf.estimator.LinearRegressor(feature_columns=['Frquency','Angle_of_Attack','Chord_Length','Free_stream_velocity','Displacement'])

# Train the estimator
#train_input = tf.estimator.inputs.numpy_input_fn(
 #   x=dict(feat_columns),
  #  y=sound_pressure_level, shuffle=False, num_epochs=None)

lin_reg.train(fileops.make_dataset(2500, feat_columns,sound_pressure_level, False, 1000),steps=2500)


print('finished')


