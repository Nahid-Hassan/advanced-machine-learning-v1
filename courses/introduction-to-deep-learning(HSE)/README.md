# Introduction to Deep Learning

## Table of Contents

- [Introduction to Deep Learning](#introduction-to-deep-learning)
  - [Table of Contents](#table-of-contents)
    - [Introduction to optimization](#introduction-to-optimization)
      - [Course Intro](#course-intro)
      - [Linear Regression](#linear-regression)
      - [Gradient Descent](#gradient-descent)
      - [Overfitting problem and model validation](#overfitting-problem-and-model-validation)
      - [Model regularization](#model-regularization)
      - [SGD](#sgd)

### Introduction to optimization

#### Course Intro

Deep learning is a fast-growing field of AI focused on using neural networks for complex practical problems. Deep neural networks are used nowadays for `object recognition` and `image analysis`, for various modules of `self-driving cars`, for `chatbots` and `natural language understanding` problems.


**Prerequisites**:

- **Linear regression**: `mean squared error`, `analytical solution`.
- **Logistic regression**: `model`, `cross-entropy loss`, `class probability estimation`.
- **Gradient descent** for `linear models`. `Derivatives` of `MSE` and `cross-entropy loss` functions.
- The problem of **overfitting**.
- **Regularization** for linear models.

[Colab Link for Testing Code](https://colab.research.google.com/drive/1nfRzTqA7DyXVNyrGiO6HIIsukk6RTVaN)

#### Linear Regression

**Regression**:

If `y` is `real number` - is regression task

- Salary prediction
- Movie Rating Problem

If `y` is belongs to a `finite set` - is classification task

- Object recognition
- Topic classification

**Linear Model**:

  𝑎(𝑥) = 𝑏 + 𝑤1𝑥 + w2x + ... + w3x

- 𝑤(,...,𝑤+—coefficients (weights)
- `𝑏` — bias
- `𝑑+1` parameters
- To make it simple: there’s always a constant feature

#### Gradient Descent

- Easy to **implement**
- Very general, can be applied to any **differentiable** loss function
- Requires less memory and **computations** (for stochastic methods)
- Gradient descent provides a `general learning framework`
- Can be used both for **classification** and **regression** tasks..


`./resources/w1_2_3_gradient.pdf`

#### Overfitting problem and model validation

- **Validation**
  - Training Set
  - Holdout Set
- **Cross Validation**
  - k-fold cross validation(If `k = 5`, we train our model 5 times.)

**Note**: Cross validation is useful for small dataset. 

**Summary**
- Models can easily overfit with high number of **parameters**
- Overfitted model just `remembers target` values for training set and `doesn’t generalize`
- Holdout set or cross-validation can be used to estimate model performance on new data.

`./resources/w1_3_1_overfit.pdf`

#### Model regularization

Reduce complexity so our model don't overfit.

- L2 Penalty
- L1 Penalty
- Dimensionality Reduction
- Data Augmentation
- Dropout
- Early Stopping
- Collect more data.

`./resources/w1_3_2_regularization.pdf`

#### SGD

`./resources/w1_4_1_sgd.pdf`
`./resources/w1_4_2_sgd.pdf`