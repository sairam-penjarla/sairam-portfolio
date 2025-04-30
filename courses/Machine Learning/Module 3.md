# Module 3: Decision Trees

Decision trees are a type of supervised learning algorithm used for both classification and regression. They work by splitting data into subsets based on feature values to make predictions.

---

## 3.1 What is a Decision Tree?

A decision tree splits the dataset into branches (decisions) until it reaches a leaf node (final decision).

**Key Concepts:**

- **Root Node**: The initial decision point.
- **Branches**: Represent decisions based on conditions.
- **Leaf Nodes**: Contain the final output or class.

---

### Code Example 1: Generating a Dataset

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print("Feature Data:")
print(X.head())
print("\\nTarget Labels:")
print(y.head())

```

**Output Explanation:**

- The Iris dataset has features like sepal length/width and petal length/width.
- Target labels represent three species of flowers.

---

### Code Example 2: Training a Decision Tree Model

```python
from sklearn.tree import DecisionTreeClassifier

# Model initialization and training
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X, y)

print("Feature Importances:")
print(dict(zip(iris.feature_names, dt_model.feature_importances_)))

```

**Output Explanation:**

- The `max_depth` parameter limits the depth of the tree to avoid overfitting.
- Feature importance indicates which features influence the decisions most.

---

### Code Example 3: Visualizing the Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

```

**Output Explanation:**

- The tree shows splits, conditions, and predictions at each node.

---

### Code Example 4: Evaluating the Model

```python
from sklearn.metrics import accuracy_score, classification_report

# Predictions
y_pred = dt_model.predict(X)

print(f"Accuracy: {accuracy_score(y, y_pred)}")
print("\\nClassification Report:")
print(classification_report(y, y_pred))

```

**Output Explanation:**

- Accuracy and classification reports help evaluate the model’s performance.