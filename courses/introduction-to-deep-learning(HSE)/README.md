# Introduction to Deep Learning

## Table of Contents

- [Introduction to Deep Learning](#introduction-to-deep-learning)
  - [Table of Contents](#table-of-contents)
    - [Introduction to optimization](#introduction-to-optimization)
      - [Course Intro](#course-intro)
      - [Linear Regression](#linear-regression)

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
