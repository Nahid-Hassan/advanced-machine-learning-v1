# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## Table of Contents
- [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](#introduction-to-tensorflow-for-artificial-intelligence-machine-learning-and-deep-learning)
  - [Table of Contents](#table-of-contents)
    - [A New Programming Paradigm](#a-new-programming-paradigm)
      - [The `Hello World` of neural networks](#the-hello-world-of-neural-networks)
      - [Housing Prices](#housing-prices)

### A New Programming Paradigm

Install **Tensorflow 2.0**

```console
!pip install tensorflow==2.0.0-alpha0 
```

#### The `Hello World` of neural networks


Map between **xs** to **ys**,

```text
xs = -1.0,  0.0, 1.0, 2.0, 3.0, 4.0
ys = -3.0, -1.0, 1.0, 3.0, 5.0, 7.0
```

```py
# import module
import tensorflow as tf
import numpy as np
from tensorflow import keras

# create model
model = keras.Sequential(keras.layers.Dense(units=1, input_shape=[1]))

# compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# prepare data
xs = np.array([-1,0,1,2,3,4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# train or fit model
model.fit(x=xs, y=ys, epochs=500)

# print model weights
print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 1) dtype=float32, numpy=array([[1.9966141]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32, numpy=array([-0.98950255], dtype=float32)>]

# predict new data
print(model.predict([12.0]))
# [[22.969868]]
```

#### Housing Prices

In this exercise you'll try to build a **neural network** that **predicts** the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs `50k + 50k` per **bedroom**, so that a `1` bedroom house costs `100k`, a `2` bedroom house costs `150k` etc.

How would you create a neural network that learns this relationship so that it would predict a `7` bedroom house as costing close to `400k` etc.

**Hint**: Your network might work better if you **scale** the house price down. You don't have to give the answer `400`...it might be better to create something that predicts the number `4, and then your answer is in the '**hundreds** of **thousands**' etc.

```py
# import module
import tensorflow as tf
import numpy as np
from tensorflow import keras

# create model
model = keras.Sequential(keras.layers.Dense(units=1, input_shape=[1]))

# compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# prepare data
xs = np.array([1, 2, 3, 4, 5], dtype=float)
ys = np.array([100, 150, 200, 250, 300], dtype=float)

# scale `ys`
ys = ys / 100
print(ys)

# train or fit model
model.fit(x=xs, y=ys, epochs=500)

# print model weights
print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 1) dtype=float32, numpy=array([[0.5058017]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32, numpy=array([0.4790542], dtype=float32)>]

# predict new data
print('The predicted housing price in(100$) :', model.predict([7.0])[0][0])
# The predicted housing price in(100$) : 4.0196657

```