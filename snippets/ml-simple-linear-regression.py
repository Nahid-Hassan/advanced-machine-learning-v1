import pandas as pd
import math
import matplotlib.pyplot as plt

# Setup dataset
df = pd.read_csv('./../code-colab/datasets/archive/Admission_Predict.csv')

# The above dataset has
# (Serial No., GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research, Chance_of_Admit) columns

print(df.head(3))

# independent variable
x = df.CGPA.values
y = df.Chance_of_Admit.values

# print(len(x), len(y))

# plot x and y and visualize the relationship
plt.scatter(x, y, marker='.')
plt.show()

# define cal_mean function
def cal_mean(data_points):
    return sum(data_points) / len(data_points)

x_mean = cal_mean(x)
y_mean = cal_mean(y)

print('x_mean:', x_mean, 'y_mean:', y_mean)

# our equation is y = b0 + b1 * x

# b1 = sum((x - x_mean) (y - y_mean)) / sum((x - x_mean) ** 2)
# b0 = y_mean - (b1 * x_mean)

def cal_b1(data_points, x_mean, y_mean):

    total_numerator = 0
    total_denominator = 0

    # sum
    for x, y in data_points:
        # sum((x - x_mean) (y - y_mean))
        total_numerator += (x - x_mean) * (y - y_mean)
        # sum((x - x_mean) ** 2)
        total_denominator += (x - x_mean) ** 2

    # sum((x - x_mean) (y - y_mean)) / sum((x - x_mean) ** 2)
    return total_numerator / total_denominator

def cal_b0(b1, x_mean, y_mean):
    # y_mean - (b1 * x_mean)
    return y_mean - (b1 * x_mean)


b1 = cal_b1(zip(x, y), x_mean, y_mean)
b0 = cal_b0(b1, x_mean, y_mean)

print('b1:', b1, 'b0', b0)

# calculate regession line based on x data points
estimates = [b0 + b1 * data for data in x]

plt.scatter(x, y, marker='.')
plt.plot(x, estimates)
plt.show()

def cal_r_square(data_points, estimates, y_mean):
    total_numerator = 0
    total_denominator = 0

    for i, y in enumerate(data_points):
        total_numerator += (estimates[i] - y_mean) ** 2
        # estimates[i] -> i'th estimate value
        total_denominator += (y - y_mean) ** 2

    return  round(total_numerator / total_denominator, 3)

r_square = cal_r_square(y, estimates, y_mean)
print('R^2:', r_square)


def cal_standard_error_of_estimate(data_points, estimates):
    total = 0
    for i, y in enumerate(data_points):
        total += (estimates[i] - y) ** 2
    return math.sqrt(total / (len(data_points) - 2))


std_error = cal_standard_error_of_estimate(y, estimates)
print("STD Error:", std_error)
