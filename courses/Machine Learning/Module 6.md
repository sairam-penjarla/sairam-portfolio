# Module 6: k-Nearest Neighbors (kNN)

## What is k-Nearest Neighbors (kNN)?

k-Nearest Neighbors (kNN) is one of the simplest and most intuitive machine learning algorithms used for both classification and regression tasks. Instead of learning an explicit model during training, kNN stores the entire dataset and uses it to make predictions during inference. This is why it is called a "lazy learner."

### Key Concepts:

1. **Instance-Based Learning**: kNN does not create an abstract model. Instead, it relies directly on the training data for predictions.
2. **Similarity Measure**: It calculates the similarity or distance between the input sample and all training samples using distance metrics like:
    - Euclidean Distance (most common)
    - Manhattan Distance
    - Minkowski Distance
3. **k**: The number of nearest neighbors to consider for making a decision.
    - For classification, the majority class among the k neighbors determines the predicted class.
    - For regression, the average value of the k neighbors determines the prediction.
4. **Decision Boundary**: The decision boundary for kNN can be non-linear and highly adaptive to the dataset, depending on the value of **k** and the distribution of data.

### Advantages:

- Simple to understand and implement.
- Performs well with smaller datasets.
- Non-parametric: It makes no assumptions about the underlying data distribution.

### Disadvantages:

- Computationally expensive as it requires computing distances for all samples in the dataset during prediction.
- Sensitive to irrelevant features and noisy data.
- Requires careful selection of **k** and distance metrics.

---

## Code Example 1: Training a kNN Classifier

### Importing Libraries and Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize kNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

```

### Explanation:

- `n_neighbors=3`: The model considers the 3 nearest neighbors for prediction.
- The accuracy score gives the percentage of correct predictions on the test set.

---

## Code Example 2: Experimenting with Different Values of k

```python
import matplotlib.pyplot as plt

# Testing multiple values of k
k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

# Plotting accuracy vs k
plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.show()

```

### Explanation:

- This plot shows how accuracy varies with the value of **k**.
- Too small or too large values of **k** can lead to underfitting or overfitting.

---

## Code Example 3: Using a Different Distance Metric

```python
# Using Manhattan Distance
knn_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_manhattan.fit(X_train, y_train)

# Make predictions
y_pred_manhattan = knn_manhattan.predict(X_test)

print(f"Accuracy with Manhattan Distance: {accuracy_score(y_test, y_pred_manhattan):.2f}")

```

### Explanation:

- The distance metric is set to `manhattan`, which computes the absolute distance between points.
- Different metrics can yield different results based on the dataset characteristics.

---

## Code Example 4: Visualizing Decision Boundaries

```python
import numpy as np

# Selecting only two features for visualization
X_2d = X_train[:, :2]
y_2d = y_train

# Train kNN with 2 features
knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_2d, y_2d)

# Create a mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict for each point in the mesh grid
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, edgecolor='k', cmap='coolwarm')
plt.title("Decision Boundary for kNN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

```

### Explanation:

- The decision boundary shows how kNN classifies regions in the feature space.
- The boundary adapts to the dataset based on the chosen **k** and distance metric.