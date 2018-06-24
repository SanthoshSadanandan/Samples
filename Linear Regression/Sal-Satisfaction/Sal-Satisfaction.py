import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.Session()

data = pd.read_csv('dataset.csv')
independent_variable = data['Salary'].values.tolist()
dependent_variable = data['Satisfaction'].values.tolist()

W = tf.Variable([3], dtype=np.float32)
b = tf.Variable([5], dtype=np.float32)
x = tf.placeholder(dtype=np.float32)

linear_model = W*x+b

act_y =  tf.placeholder(dtype=np.float32)

squared_error = tf.square(linear_model - act_y)

loss = tf.reduce_sum(squared_error)

optimizer_train = tf.train.GradientDescentOptimizer(0.000000009)

train = optimizer_train.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1,1000):
    sess.run(train, {x: independent_variable, act_y: dependent_variable})
    print('-----------------loss>>', sess.run(loss, {x: independent_variable, act_y: dependent_variable}))
    print('W=', sess.run(W))
    print('b=', sess.run(b))



print('finished')



