# import dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

# set logger
# get logger from tensorflow
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Setup training data
# feature
celsius = np.array(np.random.randint(100, size=(100)),  dtype=float)
# label

fahrenheit = np.array(list(map(lambda x: ((x * 1.8) + 32) , celsius)),  dtype=float)

# both celsius and fahrenheit makes an example. [0, 32] is an example

# create model
# build layer

# units = 1 means 1 neuron
# input_shape=[1] means This specifies that the input to this layer is a single value. That is, the shape is a one-dimensional array with one member. Since this is the first (and only) layer, that input shape is the input shape of the entire model. The single value is a floating point number, representing degrees Celsius.
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# assemble layers into model
model = tf.keras.Sequential([l0])

# display model summary
print(model.summary())

# compile the model with loss and optimizer function
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# fit or train the model

history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

# make prediction
print(model.predict([100])) # actual result is 100 * 1.8 + 32 = 212

# print weights
print(l0.get_weights())

# display loss graph
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.plot(history.history['loss'])
plt.show()

