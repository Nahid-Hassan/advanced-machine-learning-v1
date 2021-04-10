# Introduction to Deep Learning

## Table of Contents

- [Introduction to Deep Learning](#introduction-to-deep-learning)
  - [Table of Contents](#table-of-contents)
    - [Introduction to optimization](#introduction-to-optimization)
      - [Course Intro](#course-intro)
      - [Linear Regression](#linear-regression)
      - [Linear Classification](#linear-classification)

### Introduction to optimization

#### Course Intro

This is an advanced Course. So we assume that you have basic knowledge of

- Machine Learning
- Probability Theory
- Linear Algebra and Calculus
- Python Programming

**Outline**:

- [ ] : Linear Models(**Separate Lines**)
- [ ] : Multilayer Perceptron(Simplest Neural Network) - **Separate spiral lines**
- [ ] : Feed Images To Network(**CNN**)
- [ ] : **Neural Networks** (We need to understand that a NN can convert an object to a small dense vector)
- [ ] : **RNN**(Generate human names)
- [ ] : **Image captioning**

#### Linear Regression

Supervised Learning

```text
xi = example
yi = target value
xi = (xi1, xi2, ....., xid) - features
X = ((x1, y1), (x2, y2), ... , (xl, yl)) - training set
a(x) = model, hypothesis

             x ------> a(x) ------>y_pred
```

There are two main classes in supervised learning problems,

- Regression &

```text
yi E R - regression task # R is real number, yi is prediction/target value

* Salary prediction
* Movie rating prediction
```

- Classification

```text
yi belongs to a finite set - classification task

* Object Recognition(car, dog, bi-cycle) - Here object is finite.
* Topic classification (Analyze news articles)
```

**Linear Model for Regression**:

This is a simple **linear model**

```text
a(x) = b + w1x1 + w2x2 + .... + wdxd

* w1, w2, ..., wd - coefficient(weights)
* b - bias
* d + 1 parameters
* To make it simple: there's always a constant feature

Vector notation:
a(x) = wTx (dot product of weight vector)

for sample X:
a(X) = Xw

X = ((x11, ....., x1d),
     (...............),
     (xl1, ...... xld))
```

Python Code for **Dot Product**

```py
def dot_product(weights, samples):
    res = 0
    if len(weights) == len(samples):
        for i in range(len(weights)):
            res += weights[i] * samples[i]
        return res
    else:
        return 'Length of weights and samples is not matched.'

weights = [1,2,3,2,4]
samples = [2,3,1,4,1]

res = dot_product(weights, samples)
print(res)
# 23

import numpy as np
np.dot(weight, samples)
# 23

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
np.dot(a,b)
# [[37  40], [85  92]]
```

**Loss Function**:

```text
How to measure model quality?

  -- Mean Squared Error or (MSE) --

  L(w) = 1/l * sum{i=1 to l}(wTxi - yi) ^ 2
       = 1/l * || Xw - y|| ^ 2

      # Xw -> features and weights dot product
```

Python code for **MSE**

```py
import numpy as np

# Given values
Y_true = [1,1,2,2,4] # Y_true = Y (original values)

# Calculated values
Y_pred = [0.6,1.29,1.99,2.69,3.4] # Y_pred = Y'

# Mean Squared Error
MSE = np.square(np.subtract(Y_true,Y_pred)).mean()
print(MSE)
```

**Training a Model**:

Fitting the model to training data:

```text
# Minimize with respect to 'w'
L(w) = 1/l * || Xw - y|| ^ 2 ---> w(min)
```

![images](images/1.png)

#### Linear Classification

**Binary Classification**:

```text

```
