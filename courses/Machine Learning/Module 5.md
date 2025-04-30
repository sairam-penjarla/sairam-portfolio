# Module 5: Support Vector Machines (SVM)

Support Vector Machines (SVM) are supervised algorithms used for classification and regression, relying on finding the hyperplane that best separates classes.

---

## Understanding SVM

Key Concepts:

- **Margin**: Distance between the hyperplane and nearest data points from either class.
- **Support Vectors**: Points closest to the hyperplane, influencing its position.

---

### Code Example 1: Training an SVM Classifier

```python
from sklearn.svm import SVC

# SVM Model
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X.iloc[:, :2], y)  # Using only first two features for visualization

print("Support Vectors:")
print(svm_model.support_)

```

**Output Explanation:**

- `kernel="linear"` creates a linear boundary.
- Support vectors are critical data points for the hyperplane.

---

### Code Example 2: Visualizing the Decision Boundary

```python
import numpy as np

# Create grid for plotting
xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100),
                     np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 100))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='bwr')
plt.contourf(xx, yy, Z > 0, alpha=0.2)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

```

**Output Explanation:**

- The decision boundary (black line) separates the classes.

---

### Code Example 3: Tuning Parameters

```python
svm_model_rbf = SVC(kernel="rbf", gamma=0.5, C=1, random_state=42)
svm_model_rbf.fit(X.iloc[:, :2], y)

print(f"Accuracy (RBF Kernel): {svm_model_rbf.score(X.iloc[:, :2], y)}")

```

**Output Explanation:**

- Using the RBF kernel allows for non-linear decision boundaries.