# Intro to Inferential Statistics

## Table of Contents

- [Intro to Inferential Statistics](#intro-to-inferential-statistics)
  - [Table of Contents](#table-of-contents)
    - [Estimation](#estimation)
      - [Sampling Distribution Summary](#sampling-distribution-summary)
      - [Mean of Treated Population](#mean-of-treated-population)
      - [Population Mean Vs Sample Mean](#population-mean-vs-sample-mean)
      - [Percent of Sample Mean](#percent-of-sample-mean)
      - [Approximate Margin of Error](#approximate-margin-of-error)
      - [Interval Estimate for Population Mean](#interval-estimate-for-population-mean)
      - [Confidence Interval Bounds](#confidence-interval-bounds)
      - [Exact Z Score](#exact-z-score)

### Estimation

#### Sampling Distribution Summary

![images](images/1.png)

#### Mean of Treated Population

**Point Estimation**: In statistics, point estimation involves the use of sample data to calculate a single value (known as a point estimate since it identifies a point in some parameter space) which is to serve as a "best guess" or "best estimate" of an unknown population parameter (for example, the population mean).

![images](images/2.png)

#### Population Mean Vs Sample Mean

![images](images/3.png)

#### Percent of Sample Mean

Why this result: we know sample sd = sigma / sqrt_root(n).

So we can say about 95% are laying out in 2*sigma / sqrt_root(n).

![images](images/4.png)

#### Approximate Margin of Error

```py
margin_of_error = 2 * sigma / sqrt(n)
margin_of_error
```

![images](images/5.png)

#### Interval Estimate for Population Mean

```text
40 + 2*sigma / sqrt_root(n) > 40 > 40 - 2*sigma / sqrt_root(n)
```

![images](images/6.png)

#### Confidence Interval Bounds

**Python3 Code**:

```py
def confidence_interval_bounds(mean, se, samples=None):
    """
      Approximately 95% samples are fall in this interval.
      se: standard error!
    """
    left = mean - (2 * se)
    right  = mean + (2 * se)

    return (left, right)

confidence_interval_bounds(40, 2.71)
# (34.58, 45.42)
```

![images](images/7.png)

#### Exact Z Score

```py
import scipy.stats as st

# exact z score
def exact_z_score(left, right):
    left = st.norm.ppf(.025)
    right = st.norm.ppf(.975)

    return (left, right)

left, right = exact_z_score(.025, .975)
print(left, right)
# -1.9599639845400545 1.959963984540054
```

![images](images/8.png)

**Note**: This works only for sampling distribution.
