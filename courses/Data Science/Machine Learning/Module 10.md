# Module 10: Regularization Techniques (e.g., L1/L2 Regularization)

## What is Regularization?

**Regularization** is a technique used in machine learning to prevent overfitting by adding a penalty term to the model's cost function. Overfitting happens when a model learns the details and noise of the training data to the extent that it negatively impacts the performance of the model on new data. Regularization helps to reduce the model's complexity and bias-variance trade-off.

There are two common regularization techniques:

1. **L1 Regularization (Lasso)**: Adds a penalty equal to the absolute value of the coefficients. It encourages sparsity, which means some features can be completely eliminated (coefficients set to zero).
2. **L2 Regularization (Ridge)**: Adds a penalty equal to the square of the coefficients. It prevents the model from becoming too sensitive to any one feature, leading to smoother and more generalizable models.

Both techniques are used to control the model's capacity and make it more robust by discouraging the learning of overly complex patterns in the data.

---

## Section 10.1: L1 Regularization (Lasso)

### How L1 Regularization Works:

L1 regularization adds the sum of the absolute values of the coefficients to the loss function. The formula looks like this:
\[ \text{Loss Function} = \text{Original Loss} + \lambda \sum_{i=1}^{n} |w_i| \]

Where:

- **\( \lambda \)** is the regularization parameter that controls the strength of the penalty.
- **\( w_i \)** are the model's coefficients.

L1 regularization can result in sparse models, where some coefficients become exactly zero, making it useful for feature selection.

### Code Example 1: L1 Regularization in Linear Regression

**Importing Libraries and Dataset**

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```

**Explanation**

- We generate a synthetic regression dataset using **`make_regression`**.
- The dataset is split into training and testing sets.

---

**Training a Lasso Model**

```python
# Train Lasso (L1 Regularization)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = lasso_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) with L1 Regularization:", mse)

```

**Explanation**

- **`alpha=0.1`** controls the strength of the regularization.
- The Mean Squared Error (MSE) helps to evaluate the model's performance.

---

**Code Example 2: Visualizing Feature Selection with Lasso**

```python
import matplotlib.pyplot as plt

# Plot the coefficients learned by Lasso
plt.bar(range(len(lasso_reg.coef_)), lasso_reg.coef_)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regularization Coefficients')
plt.show()

```

**Explanation**

- This plot visualizes which features have been eliminated by Lasso (coefficients set to zero) and which have been retained.

---

## Section 10.2: L2 Regularization (Ridge)

### How L2 Regularization Works:

L2 regularization adds the sum of the squared values of the coefficients to the loss function. The formula looks like this:
\[ \text{Loss Function} = \text{Original Loss} + \lambda \sum_{i=1}^{n} w_i^2 \]

Where:

- **\( \lambda \)** is the regularization parameter that controls the strength of the penalty.
- **\( w_i \)** are the model's coefficients.

Unlike L1 regularization, L2 does not result in sparse models. It keeps all features but reduces their influence by shrinking their coefficients.

### Code Example 3: L2 Regularization in Linear Regression

**Training a Ridge Model**

```python
from sklearn.linear_model import Ridge

# Train Ridge (L2 Regularization)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = ridge_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) with L2 Regularization:", mse)

```

**Explanation**

- **`alpha=1.0`** controls the strength of the regularization.
- We calculate the MSE to evaluate the model's performance.

---

**Code Example 4: Comparing L1 and L2 Regularization**

```python
# Train both Lasso (L1) and Ridge (L2)
lasso_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)

# Compare the coefficients
print("Lasso Coefficients:", lasso_reg.coef_)
print("Ridge Coefficients:", ridge_reg.coef_)

```

**Explanation**

- The coefficients of Lasso will show some features with zero weights, while Ridge will shrink all coefficients, but none will be exactly zero.

---

## Section 10.3: Elastic Net Regularization

### Combining L1 and L2 Regularization

**Elastic Net** combines the properties of both L1 and L2 regularization. It is useful when there are multiple correlated features. The cost function for Elastic Net is:
\[ \text{Loss Function} = \text{Original Loss} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2 \]

Elastic Net regularization allows for both feature selection (like Lasso) and coefficient shrinkage (like Ridge).

### Code Example 5: Elastic Net Regularization

```python
from sklearn.linear_model import ElasticNet

# Train Elastic Net
elastic_net_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = elastic_net_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) with Elastic Net Regularization:", mse)

```

**Explanation**

- **`l1_ratio=0.5`** indicates equal mixing of L1 and L2 regularization.
- **Elastic Net** provides a balance between Lasso and Ridge.