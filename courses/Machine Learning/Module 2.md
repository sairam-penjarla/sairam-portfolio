# Module 2: Logistic Regression

Logistic regression is used for binary classification problems, such as predicting whether an email is spam or not.

## Understanding Logistic Regression

Unlike linear regression, logistic regression predicts probabilities using the sigmoid function:

**Sigmoid Function:**

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

Where \( z = \beta_0 + \beta_1 x \).

---

### Code Example 1: Generating a Classification Dataset

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, n_features=1, n_classes=2, random_state=42)

plt.scatter(X, y, c=y, cmap='bwr', label='Data Points')
plt.xlabel('Feature')
plt.ylabel('Class')
plt.title('Generated Binary Classification Data')
plt.legend()
plt.show()

```

**Output Explanation:**

- Red and blue dots represent two classes.

---

### Code Example 2: Training a Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

# Model initialization and training
log_model = LogisticRegression()
log_model.fit(X, y)

# Model parameters
print(f"Intercept: {log_model.intercept_[0]}")
print(f"Coefficient: {log_model.coef_[0][0]}")

```

**Output:**

- The model learns parameters to separate the two classes.

---

### Code Example 3: Predicting and Visualizing

```python
import numpy as np

# Predictions
X_new = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = log_model.predict_proba(X_new)[:, 1]

# Plotting
plt.scatter(X, y, c=y, cmap='bwr', label='Data Points')
plt.plot(X_new, y_prob, color='green', label='Prediction Probability')
plt.xlabel('Feature')
plt.ylabel('Probability of Class 1')
plt.title('Logistic Regression Predictions')
plt.legend()
plt.show()

```

**Output Explanation:**

- The green curve represents the probability of belonging to class 1.

---

### Code Example 4: Evaluating the Model

```python
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = log_model.predict(X)

print(f"Accuracy: {accuracy_score(y, y_pred)}")
print(f"Confusion Matrix:\\n{confusion_matrix(y, y_pred)}")

```

**Output:**

- **Accuracy:** Measures overall correctness.
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.