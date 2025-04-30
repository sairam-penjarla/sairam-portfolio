# Module 9: Ensemble Methods (e.g., Bagging, Boosting)

## What are Ensemble Methods?

**Ensemble methods** are machine learning techniques that combine predictions from multiple models to improve accuracy, robustness, and generalizability. The idea is to leverage the strengths of multiple models to minimize errors or biases that might occur when using a single model.

Ensemble methods are categorized into two main types:

1. **Bagging (Bootstrap Aggregating)**: Reduces variance by training multiple models independently and averaging their outputs.
2. **Boosting**: Reduces bias by training models sequentially, where each model focuses on correcting the errors of its predecessor.

### Why Use Ensemble Methods?

- Improved accuracy and performance.
- Robustness to overfitting compared to individual models.
- Handles complex data distributions effectively.

---

## Section 9.1: Bagging (Bootstrap Aggregating)

### How Bagging Works:

1. Create multiple subsets of the training data using random sampling (with replacement).
2. Train a base model (e.g., Decision Tree) on each subset independently.
3. Combine the predictions from all models (e.g., majority voting for classification or averaging for regression).

### Code Example 1: Bagging with Decision Trees

**Importing Libraries and Dataset**

```python
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```

**Explanation**

- **`make_classification`** generates a dataset for classification tasks.
- Data is split into training and testing sets.

---

**Training a Bagging Classifier**

```python
# Train Bagging Classifier
bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = bagging_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

```

**Explanation**

- **`base_estimator=DecisionTreeClassifier()`**: Uses Decision Trees as the base model.
- **`n_estimators=10`**: Combines predictions from 10 Decision Trees.
- The accuracy score shows the performance of the bagging classifier.

---

### Code Example 2: Comparing Bagging and Single Models

**Training a Single Decision Tree**

```python
# Train a single Decision Tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

# Evaluate the single Decision Tree
single_pred = single_tree.predict(X_test)
print("Single Decision Tree Accuracy:", accuracy_score(y_test, single_pred))

```

**Explanation**

- A single Decision Tree often overfits, leading to lower accuracy compared to a Bagging ensemble.

---

### Code Example 3: Visualizing the Effect of Bagging

**Visualizing Decision Boundaries**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Create a simple dataset
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# Train Bagging Classifier
bagging_clf.fit(X, y)

# Visualize decision boundaries
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.show()

plot_decision_boundary(bagging_clf, X, y)

```

**Explanation**

- The decision boundary created by Bagging is smoother and more generalizable than a single Decision Tree.

---

## Section 9.2: Boosting

### How Boosting Works:

1. Train a weak learner (e.g., Decision Tree) on the dataset.
2. Adjust weights to focus on the misclassified examples.
3. Sequentially train new weak learners on the updated data, combining their predictions.

### Popular Boosting Algorithms:

- **AdaBoost (Adaptive Boosting)**: Assigns weights to misclassified samples and combines weak learners iteratively.
- **Gradient Boosting**: Optimizes the loss function using gradients.
- **XGBoost/LightGBM/CatBoost**: Efficient and optimized implementations of Gradient Boosting.

---

### Code Example 1: AdaBoost

**Importing Libraries and Dataset**

```python
from sklearn.ensemble import AdaBoostClassifier

# Train AdaBoost Classifier
adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = adaboost_clf.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))

```

**Explanation**

- **`n_estimators=50`**: Combines 50 weak learners.
- AdaBoost improves performance by focusing on difficult-to-classify samples.

---

### Code Example 2: Gradient Boosting with sklearn

**Training Gradient Boosting Classifier**

```python
from sklearn.ensemble import GradientBoostingClassifier

# Train Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = gb_clf.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))

```

**Explanation**

- **`learning_rate=0.1`**: Controls the contribution of each weak learner.
- Gradient Boosting effectively minimizes the loss function.

---

### Code Example 3: XGBoost

**Training an XGBoost Classifier**

```python
from xgboost import XGBClassifier

# Train XGBoost Classifier
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_clf.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

```

**Explanation**

- **XGBoost** is a faster and more efficient Gradient Boosting implementation.
- Its scalability makes it ideal for large datasets.

---

## Summary of Ensemble Methods

| Method | Description | Key Use Case |
| --- | --- | --- |
| Bagging | Combines predictions from models trained independently | Reduces variance |
| Boosting | Sequentially trains models to correct errors | Reduces bias |
| Gradient Boosting | Optimizes loss function using gradients | High performance on complex data |

Ensemble methods are a cornerstone of modern machine learning. Try using **PyCharm** or **VSCode** to experiment with Bagging and Boosting on different datasets. Hands-on practice will deepen your understanding!