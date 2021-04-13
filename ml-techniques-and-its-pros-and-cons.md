# Machine Learning Techniques and Its Pros and Crons

## Table of Contents

- [Machine Learning Techniques and Its Pros and Crons](#machine-learning-techniques-and-its-pros-and-crons)
  - [Table of Contents](#table-of-contents)
    - [C](#c)
      - [Cross Validation](#cross-validation)
    - [H](#h)
      - [Hyperparameters Tuning](#hyperparameters-tuning)


### C

#### Cross Validation

Cross Validation in Machine Learning is a great technique to deal with overfitting problem in various algorithms. Instead of training our model on one training dataset, we train our model on many datasets. Below are some of the advantages and disadvantages of Cross Validation in Machine Learning.

**Advantages of Cross Validation**:

- **Reduces Overfitting**: In Cross Validation, we split the dataset into multiple folds and train the algorithm on different folds. This `prevents our model from overfitting` the training dataset. So, in this way, the model attains the `generalization capabilities` which is a good sign of a robust algorithm.

**Note**: Chances of overfitting are less if the dataset is large. So, `Cross Validation may not be required at all in the situation where we have sufficient data available`.

- **Hyperparameters Tuning**: Cross Validation helps in finding the optimal value of hyperparameters to increase the efficiency of the algorithm.

**Disadvantages of Cross Validation**:

- **Increases Training Time**: Cross Validation drastically increases the training time. Earlier you had to train your model only on one training set, but with Cross Validation you have to train your model on multiple training sets.  
For example, if you go with `5 Fold` Cross Validation, you need to do 5 rounds of training each on different `4/5` of available data. And this is for only one choice of hyperparameter. If you have `multiple choice of parameters`, then the training period will shoot too high.

- **Needs Expensive Computation**: Cross Validation is computationally very expensive in terms of processing power required.

### H

#### Hyperparameters Tuning

Hyperparameters are the parameters which we pass to the Machine Learning algorithms to `maximize` their **performance** and **accuracy**. 

For example, we need to pass the optimal value of `K` in the `KNN` algorithm so that it delivers good accuracy as well as does not `underfit / overfit`. Our model should have a good generalization capability. So, choosing the optimal value of the hyperparameter is very crucial in the Machine Learning algorithms.

**Examples of Hyperparameters**:

1. K in **KNN** (Number of nearest neighbors in KNN algorithm)

2. K in **K-Means** Clustering (Number of clusters in K-Means Clustering algorithm)

3. Depth of a Decision Tree

4. Number of Leaf Nodes in a **Decision Tree**

5. Number of Trees in a **Random Forest**

6. Step Size and **Learning Rate** in Gradient Descent (or Stochastic Gradient Descent)

7. Regularization Penalty (Lambda) in **Ridge** and **Lasso Regression**.

Hyperparameters are also called "`meta-parameters`" and "`free parameters`".
