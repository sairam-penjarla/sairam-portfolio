# Module 11: Optimization Algorithms (e.g., Gradient Descent, Stochastic Gradient Descent)

## What is Optimization?

**Optimization** is a core concept in machine learning where the objective is to find the best parameters (weights) for a model that minimize or maximize a given function (typically a loss or cost function). The optimization algorithm adjusts the parameters iteratively to reduce the error (or loss) between the model's predictions and the true values.

The goal is to find the **global minimum** of the cost function, though in practice, we often settle for a **local minimum**. Optimization algorithms are used to efficiently search for these optimal parameters. The most common optimization algorithms include **Gradient Descent (GD)** and its variants, such as **Stochastic Gradient Descent (SGD)**.

---

## Section 11.1: Gradient Descent (GD)

### How Gradient Descent Works:

Gradient Descent is an iterative optimization algorithm used to minimize the cost function. The general idea is to update the model parameters in the opposite direction of the gradient (slope) of the cost function.

The update rule for each parameter is:
\[ \theta = \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta} \]

Where:

- **\( \theta \)** represents the parameters (weights).
- **\( \alpha \)** is the learning rate, which controls the step size.
- **\( J(\theta) \)** is the cost function.

The algorithm computes the gradient of the cost function with respect to the model parameters, then updates the parameters by taking a step in the opposite direction of the gradient.

### Code Example 1: Basic Gradient Descent

**Importing Libraries and Dataset**

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic linear data: y = 2x + 1
X = 2 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)  # Adding some noise

```

**Explanation**

- We generate synthetic data where the true relationship between \( X \) and \( y \) is \( y = 2x + 1 \), with some noise.

---

**Gradient Descent Function**

```python
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(X)
    theta = np.random.randn(2, 1)  # Random initialization of parameters
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
    return theta

```

**Explanation**

- **`X_b`** is the augmented feature matrix that includes the bias term (column of ones).
- The gradients are computed using the cost function's partial derivatives with respect to each parameter.
- The parameters are updated using the learning rate and gradients.

---

**Running Gradient Descent and Plotting**

```python
theta = gradient_descent(X, y)

# Plotting the data and the learned line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, X.dot(theta[1:]) + theta[0], color='red', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```

**Explanation**

- We plot the original data points (blue) and the learned line (red) to visualize the result of gradient descent.

---

## Section 11.2: Stochastic Gradient Descent (SGD)

### How Stochastic Gradient Descent Works:

Unlike regular Gradient Descent, which computes the gradient using the entire dataset, **Stochastic Gradient Descent (SGD)** updates the parameters using only a single randomly chosen data point (or a small batch). This makes SGD more computationally efficient, especially for large datasets.

The update rule for SGD is similar to GD, but the gradient is calculated for each data point (or mini-batch):
\[ \theta = \theta - \alpha \cdot \nabla J(\theta; x^{(i)}, y^{(i)}) \]

Where:

- **\( x^{(i)}, y^{(i)} \)** are the features and target values of a single data point or a mini-batch.

This can lead to faster convergence in practice, especially for large datasets, but it introduces noise in the updates.

### Code Example 2: Stochastic Gradient Descent

**SGD Function**

```python
def stochastic_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(X)
    theta = np.random.randn(2, 1)  # Random initialization of parameters
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    for iteration in range(n_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients
    return theta

```

**Explanation**

- In each iteration, we randomly pick a data point to compute the gradient and update the parameters.
- This noisy update allows the algorithm to escape local minima and converge more quickly for large datasets.

---

**Running Stochastic Gradient Descent and Plotting**

```python
theta_sgd = stochastic_gradient_descent(X, y)

# Plotting the data and the learned line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, X.dot(theta_sgd[1:]) + theta_sgd[0], color='green', label='SGD Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

```

**Explanation**

- We plot the result of SGD on the same dataset to compare how it performs relative to Gradient Descent.

---

## Section 11.3: Comparing Gradient Descent and Stochastic Gradient Descent

### Batch Gradient Descent vs Stochastic Gradient Descent

- **Batch Gradient Descent**: Uses the entire dataset to compute the gradient in each iteration. It's stable but can be slow for large datasets.
- **Stochastic Gradient Descent (SGD)**: Uses one data point at a time, which makes it faster and more efficient for large datasets but introduces noise in the optimization process.

**Advantages of SGD**:

- Faster for large datasets.
- Can escape local minima more effectively.
- Suitable for online learning.

**Disadvantages of SGD**:

- The noise in the gradient can slow down convergence, leading to oscillations.

---

**Code Example 3: Comparing Convergence of GD and SGD**

```python
theta_gd = gradient_descent(X, y)
theta_sgd = stochastic_gradient_descent(X, y)

print("Gradient Descent Coefficients:", theta_gd)
print("Stochastic Gradient Descent Coefficients:", theta_sgd)

```

**Explanation**

- This comparison shows how the coefficients learned by both algorithms may converge to similar values, but the training process and convergence speed differ.