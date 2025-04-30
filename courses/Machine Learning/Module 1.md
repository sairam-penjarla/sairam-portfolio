# Module 1: Linear Regression

Linear regression is a supervised learning algorithm for predicting a dependent variable (output) based on one or more independent variables (inputs).

## 1.1 What is Linear Regression?

Linear regression fits a line (or hyperplane) that minimizes the difference between predicted and actual values.

**Equation:**

\[ y = \beta_0 + \beta_1 x \]

Where:

- \( y \) = predicted output
- \( x \) = input feature
- \( \beta_0 \), \( \beta_1 \) = coefficients

---

## 1.2 Implementing Simple Linear Regression

Let’s predict house prices based on the size of a house.

### Code Example 1: Dataset Preparation

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulated data
np.random.seed(42)
X = np.random.rand(50, 1) * 100  # House sizes (in sq ft)
y = 5000 + 250 * X + np.random.randn(50, 1) * 20000  # Prices (with noise)

plt.scatter(X, y, color='blue', label='Data Points')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Size vs Price')
plt.legend()
plt.show()

```

**Output Explanation:**

- The scatter plot shows the relationship between house size and price.
- Notice the noise (random variation) in the data.

---

### Code Example 2: Training the Model

```python
from sklearn.linear_model import LinearRegression

# Reshape data for scikit-learn
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Model initialization and training
model = LinearRegression()
model.fit(X, y)

# Model parameters
print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficient: {model.coef_[0][0]}")

```

**Output:**

- The intercept (\( \beta_0 \)) and coefficient (\( \beta_1 \)) are displayed. These define the best-fit line.

---

### Code Example 3: Visualizing Predictions

```python
# Predictions
y_pred = model.predict(X)

# Plotting
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Best Fit Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

```

**Output Explanation:**

- The red line represents the model's predictions. It minimizes the error between predicted and actual prices.

---

### Code Example 4: Evaluating the Model

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

```

**Output:**

- **Mean Squared Error (MSE):** Measures average prediction error.
- **R-squared:** Indicates how well the model explains the variance in data (closer to 1 is better).

---

## 1.3 Key Takeaways

1. Linear regression works well for linear relationships.
2. MSE and R-squared help assess model performance.
3. Hands-on practice reinforces understanding—experiment with different datasets!