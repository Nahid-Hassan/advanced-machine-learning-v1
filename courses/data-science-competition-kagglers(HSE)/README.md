# How to Win a Data Science Competition: Learn from Top Kagglers

## Table of Contents

- [How to Win a Data Science Competition: Learn from Top Kagglers](#how-to-win-a-data-science-competition-learn-from-top-kagglers)
  - [Table of Contents](#table-of-contents)
    - [Week 1 Overview](#week-1-overview)
    - [Competition Mechanics](#competition-mechanics)
      - [Recap of Machine Learning Algorithm](#recap-of-machine-learning-algorithm)
      - [Hardware and Software](#hardware-and-software)
      - [Numeric Futures](#numeric-futures)

### Week 1 Overview

Welcome to the first week of the "How to Win a Data Science Competition" course! This is a short summary of what you will learn. 

- **Mikhail Trofimov**: will introduce you to competitive data science. You will learn about competitions' mechanics, the difference between **competitions** and a **real-life data science**, overview **hardware** and **software** that people usually use in competitions. We will also briefly recap major ML models frequently used in competitions. 
- **Alexander Guschin**: will summarize approaches to work with **features**: `preprocessing`, `generation` and `extraction`. We will see, that the choice of the machine learning model impacts both preprocessing we apply to the features and our approach to generation of new ones. We will also discuss feature extraction from text with Bag Of Words and **Word2vec**, and feature extraction from images with **Convolution Neural Networks**.

### Competition Mechanics

**Data**:  Data is what the organizers give us as training material. We will use it in order to produce our solution. Data can be represented in a variety of formats. **CSV** file with several columns , a **text** file, an archive with **pictures**, a **database dump**, a **disabled** code or even all together. With the data, usually there is a description. It's useful to read it in order to understand what we'll work with and which feature can be extracted.

**Model**:  This is exactly what we will build during the competition. It's better to think about model not as one specific algorithm, but something that `transforms data into answers`. The model should have **two** main **properties**. It should produce **best possible prediction** and be **reproducible**. In fact, it can be very complicated and contain a lot of algorithms, handcrafted features, use a variety of libraries as this model of the winners of the Homesite competition shown on this slide. It's large and includes many components.

**Submission**: To compare our model with the model of other participants, we will send our predictions to the server or in other words, make the submission. Usually, you're asked about predictions only. Sources or models are not required. And also there are some exceptions, cool competitions, where participants submit their code. In this course, we'll focus on traditional challenges where a competitor submit only prediction outputs. Often, I can not just provide a so-called sample submission. An example of how the submission file should look like, look at the sample submission from the **Zillow** competition. In it is the first column. We must specify the ID of the object and then specify our prediction for it. This is typical format that is used in many competitions.

**Evaluation**: When you submit predictions, you need to know how good is your model. The quality of the model is defined by evaluation function. In essence and simply the function, the text prediction and correct answers and returns a score characterizes the performance of the solution. The simplest example of such a function is the accurate score. This is just a rate of correct answers. In general, there are a lot of such functions. In our course, we will carefully consider some of them. The description of the competition always indicates which evaluation function is used. I strongly suggest you to pay attention to this function because it is what we will try to optimize. 

`Evaluation Functions`: **Accuracy**, **Logistic Loss**, **AUC**, **RMSE**, **MAE**. 

**Leaderboards**: The ranking of your kaggle competition.


#### Recap of Machine Learning Algorithm

- **Linear Model**: `Logistic Regression`, `SVM`
    Packages: Scikit-Learn, vowpal-wabblt(for large dataset)
- **Tree Based**: `Decision Tree`, `Random Forest`, `GBDT(Gradient Boaster Decision Tree)`
    Packages: Scikit-Learn, XGBoost(faster, dmlc), lightBGM(faster, microsoft) 

- **KNN**: Scikit-Learn(Allow your own custom distance function.)
- **Neural Network**: Tensorflow, Keras, Pytroch(Flexible), Lasagnes.

> **Note**: XGBoost and Neural Networks is awesome but don't underestimate others.

**Overview of methods**:

- Scikit-Learn (or sklearn) library
- Overview of k-NN (sklearn's documentation)
- Overview of Linear Models (sklearn's documentation)
- Overview of Decision Trees (sklearn's documentation)
- Overview of algorithms and parameters in H2O documentation

**Additional Tools**:

- Vowpal Wabbit repository
- XGBoost repository
- LightGBM repository
- Interactive demo of simple feed-forward Neural Net
- Frameworks for Neural Nets: Keras,PyTorch,TensorFlow,MXNet, Lasagne
- Example from sklearn with different decision surfaces
Arbitrary order factorization machines

#### Hardware and Software

**Most Competition Except (Image Classification)**:

- High Level Laptop
- 16GB+ RAM
- 4+ Cores

**Quite Good Setup**:

- Tower PC
- 32GB+ RAM
- 6+ Cores


**Additional Material and Links**:

**StandCloud Computing**:

- AWS, 
- Google Cloud 
- Microsoft Azure

**AWS spot option**:

- Overview of Spot mechanism
- Spot Setup Guide

**Stack and packages**:

- Basic SciPy stack (ipython, numpy, pandas, matplotlib)
- Jupyter Notebook
- Stand-alone python tSNE package
- Libraries to work with sparse CTR-like data: LibFM, LibFFM
- Another tree-based method: RGF (implementation, paper)
- Python distribution with all-included packages: Anaconda
- Blog "datas-frame" (contains posts about effective Pandas usage)

#### Numeric Futures

Basic approach as to `feature preprocessing` and `feature generation` for **numeric** features.

**Feature Scale** are important for Non-tree based Algorithm like
- KNN
- Linear Model
- Linear SVM, SVM
- and Neural Network.

1. **Preprocessing.scaling**

**Example - 1**:

```py
# To[0,1]

from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([10, 20, 30])
print(X)

# scaling
for m in X:
    print((m - X.min()) / (X.max() - X.min()))
```

**Example - 2**:

```py
data = pd.read_csv('./datasets/titanic_train.csv')

data[['Age', 'SibSp', 'Fare']].hist(figsize=(10, 4))
xtrain = scalar.fit_transform(data[['Age', 'SibSp', 'Fare']])
pd.DataFrame(xtrain).hist(figsize=(10,5))
```

> **Note**: We use preprocessing to scale all features to one scale, so that their initial impact on the model will be roughly similar. For example, as in the recent example where we used KNN for prediction, this could lead to the case where some features will have critical influence on predictions.

**Standard Scalar**:

```py
# X is numpy array.
# x is a element of X
x = (x - X.mean() / x.std())
```

2. **Outliers**:

**Clip** features values between two chosen values of `lower bound` and `upper bound`. We can choose them as some **percentiles** of that feature. For example, `1st` and `99st` percentiles. This procedure of clipping is well-known in financial data and it is called **winsorization**. 

```py
# 1 to 99 percentile
# return lower and upper bound value
upper_bound, lower_bound = np.percentile(x, [1, 99])
y = np.clip(x, upper_bound, lower_bound)
pd.Series(y).hist(bins=30)
```

3. **Rank**:

If we apply a rank to the source of array, it will just change values to their indices. Now, if we apply a rank to the not-sorted array, it will sort this array, define mapping between values and indices in this source of array, and apply this mapping to the initial array. Linear models, KNN, and neural networks can benefit from this kind of transformation if we have no time to handle outliers manually. Rank can be imported as a random data function from scipy. One more important note about the rank transformation is that to apply to the test data, you need to store the creative mapping from features values to their rank values. Or alternatively, you can concatenate, train, and test data before applying the rank transformation.

```py
from scipy.stats import rankdata

x = [1,2,3,4,2,3]
print(rankdata(x))
# [1.  2.5 4.5 6.  2.5 4.5]
```
