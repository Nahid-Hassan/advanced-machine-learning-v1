# Intro to Descriptive Statistics

Mathematics for Understanding Data. Statistics is an important field of math that is used to analyze, interpret, and predict outcomes from data. Descriptive statistics will teach you the basic concepts used to describe data. This is a great beginner course for those interested in Data Science, Economics, Psychology, Machine Learning, Sports analytics and just about any other field.

## Table of Contents

- [Intro to Descriptive Statistics](#intro-to-descriptive-statistics)
  - [Table of Contents](#table-of-contents)
    - [Visualizing Data](#visualizing-data)
      - [Frequency](#frequency)

### Visualizing Data

#### Frequency

In statistics the frequency (or absolute frequency) of an event ***n<sub>i</sub>*** is the number ***n<sub>i</sub>*** of times the observation occurred/recorded in an experiment or study. These frequencies are often graphically represented in `histograms`.

**Types**:

The `cumulative frequency` is the total of the absolute frequencies of all events at or below a certain point in an ordered list of events.

The `relative frequency` (or `empirical probability`) of an event is the absolute frequency normalized by the total number of events:

![Relative Frequency](https://wikimedia.org/api/rest_v1/media/math/render/svg/61bca88bb54862d881e92de20c5fa6a4c5626df1)

The values of ***f<sub>i</sub>*** for all events ***i*** can be plotted to produce a frequency distribution.

In the case when ***n<sub>i</sub> = 0*** for certain ***i***, pseudocounts can be added.

```py
from collections import Counter

fruits = "apple banana apple strawberry banana lemon"
fruits = fruits.split()

fruits_count = Counter(fruits) # take list and return frequency dictionary
for key, values in fruits_count.items():
    print(key, values)
```
