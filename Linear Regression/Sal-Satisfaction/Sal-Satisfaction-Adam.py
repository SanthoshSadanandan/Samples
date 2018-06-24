import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('dataset.csv')
independent_variable = data['Salary'].values
dependent_variable = data['Satisfaction'].values

# Create feature column and estimator
column = tf.feature_column.numeric_column('x')
lin_reg = tf.estimator.LinearRegressor(feature_columns=[column])

# Train the estimator
train_input = tf.estimator.inputs.numpy_input_fn(
    x={"x": independent_variable},
    y=dependent_variable, shuffle=False,num_epochs=None)

lin_reg.train(train_input,steps=2500)

# Make two predictions
predict_input = tf.estimator.inputs.numpy_input_fn(
     x={"x": np.array([40, 60, 80, 100, 20], dtype=np.float32)},
     num_epochs=1, shuffle=False)
results = lin_reg.predict(predict_input)

 # Print result
for value in results:
     print(value['predictions'])

print('finished')



