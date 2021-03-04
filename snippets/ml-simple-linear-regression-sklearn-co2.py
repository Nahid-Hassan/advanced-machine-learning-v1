from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read FuelConsumption.csv
df = pd.read_csv('./../code-colab/datasets/FuelConsumption.csv')

# Remove/Drop Nan cell/row
df.dropna(inplace=True)

# describe the dataset
print(df.describe())

# create new dataframe for more expolore
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# visualize dataset
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
plt.xlabel("Engine Size")
plt.ylabel('CO2 Emissions')
plt.show()

# Split dataset
mask = np.random.rand(len(cdf)) < .8

train = cdf[mask]
test = cdf[~mask]

# create model using sklearn.linear_model
from sklearn import linear_model
regr = linear_model.LinearRegression()

train_x = np.asanyarray(cdf[['ENGINESIZE']])
train_y = np.asanyarray(cdf[['CO2EMISSIONS']])

regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)

# Draw regression line
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
# -g for green line
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-g')
plt.xlabel("Engine Size")
plt.ylabel('CO2 Emissions')
plt.show()

# evaluate
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

# less is better
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

# high is better, maximum: 1.0
print("R2-score: %.2f" % r2_score(test_y, test_y_) )
