> Data preprocessing is the backbone of every successful machine learning project. It involves cleaning, transforming, and preparing data to improve model performance. This guide is divided into multiple modules, each covering specific preprocessing techniques with Python code snippets, explanations, and outputs.
> 

# Table of contents

## Module 1: Data Cleaning

### 1.1 Handling Missing Values

### 1.1.1 Using `pandas.DataFrame.fillna()`

```python
import pandas as pd

# Sample DataFrame with missing values
data = {'Name': ['Alice', 'Bob', None],
        'Age': [25, None, 30],
        'Score': [85, 90, None]}
df = pd.DataFrame(data)

# Fill missing values with specified values
df_filled = df.fillna({'Age': df['Age'].mean(), 'Score': 0})
print("DataFrame with Filled Missing Values:")
print(df_filled)

```

**Explanation:**

- The `fillna()` function replaces missing values with specific values (e.g., mean for numerical columns, default for categorical columns).

**Output:**

```
      Name   Age  Score
0    Alice  25.0   85.0
1      Bob  27.5    0.0
2     None  30.0    0.0

```

---

### 1.1.2 Using `pandas.DataFrame.dropna()`

```python
# Drop rows with missing values
df_dropped = df.dropna()
print("DataFrame with Dropped Missing Values:")
print(df_dropped)

```

**Explanation:**

- `dropna()` removes rows or columns that contain missing data.

**Output:**

```
   Name   Age  Score
0 Alice  25.0   85.0

```

---

### 1.1.3 Using `sklearn.impute.SimpleImputer`

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Sample data with missing values
data = [[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
print("Data After Imputation:")
print(data_imputed)

```

**Explanation:**

- `SimpleImputer` provides a robust way to handle missing values using strategies like `'mean'`, `'median'`, and `'most_frequent'`.

**Output:**

```
[[1. 2. 7.5]
 [4. 5. 6. ]
 [7. 8. 9. ]]

```

---

### 1.2 Handling Outliers

### 1.2.1 Z-Score Method

```python
import numpy as np

# Sample data
data = np.array([10, 12, 14, 15, 18, 95])

# Calculate Z-scores
mean = np.mean(data)
std = np.std(data)
z_scores = (data - mean) / std

# Identify outliers
outliers = data[np.abs(z_scores) > 2]
print("Outliers Detected:", outliers)

```

**Explanation:**

- Z-Scores identify data points that deviate significantly from the mean.
- Data points with Z-Scores > 2 or < -2 are outliers.

**Output:**

```
Outliers Detected: [95]

```

---

### 1.2.2 IQR Method

```python
# Calculate IQR
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]
print("Outliers Detected:", outliers)

```

**Explanation:**

- IQR (Interquartile Range) is a robust method to detect outliers.
- Data outside `[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]` is considered an outlier.

**Output:**

```
Outliers Detected: [95]

```

---

### 1.2.3 Using `numpy.clip()`

```python
# Clipping outliers to a specified range
clipped_data = np.clip(data, lower_bound, upper_bound)
print("Data After Clipping:", clipped_data)

```

**Explanation:**

- `numpy.clip()` limits values within a specified range, effectively capping outliers.

**Output:**

```
Data After Clipping: [10 12 14 15 18 25]

```

---

## Module 2: Data Normalization and Scaling

### 2.1 StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# Sample data
data = [[1, 2], [3, 4], [5, 6]]

# Standard Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Standard Scaled Data:")
print(data_scaled)

```

**Explanation:**

- Scales data to have mean = 0 and standard deviation = 1.

**Output:**

```
Standard Scaled Data:
[[-1.22474487 -1.22474487]
 [ 0.          0.        ]
 [ 1.22474487  1.22474487]]

```

---

### 2.2 MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

# Min-Max Scaling
scaler = MinMaxScaler()
data_minmax = scaler.fit_transform(data)
print("Min-Max Scaled Data:")
print(data_minmax)

```

**Explanation:**

- Scales data to a fixed range, typically [0, 1].

**Output:**

```
Min-Max Scaled Data:
[[0. 0.]
 [0.5 0.5]
 [1. 1.]]

```

---

### 2.3 RobustScaler

```python
from sklearn.preprocessing import RobustScaler

# Robust Scaling
scaler = RobustScaler()
data_robust = scaler.fit_transform(data)
print("Robust Scaled Data:")
print(data_robust)

```

**Explanation:**

- Scales data using the median and IQR, making it robust to outliers.

**Output:**

```
Robust Scaled Data:
[[-1. -1.]
 [ 0.  0.]
 [ 1.  1.]]

```

---

## Coming Up Next:

- Data Transformation (Encoding, Binning, Log Transformation).
- Feature Engineering (Polynomial Features, Interaction Features).
- Feature Selection, Data Augmentation, and more.

Continue practicing and experimenting with these techniques in PyCharm or VSCode to gain a solid understanding of data preprocessing. Stay tuned for the next modules!