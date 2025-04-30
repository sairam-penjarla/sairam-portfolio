# Module 4: Random Forests

Random forests are an ensemble method that combines multiple decision trees to improve prediction accuracy.

---

## What is a Random Forest?

A random forest:

1. Creates multiple decision trees.
2. Aggregates their predictions (majority vote for classification or average for regression).

---

### Code Example 1: Training a Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

# Model initialization and training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

print("Feature Importances:")
print(dict(zip(iris.feature_names, rf_model.feature_importances_)))

```

**Output Explanation:**

- `n_estimators` specifies the number of trees.
- Feature importance shows the aggregated importance across all trees.

---

### Code Example 2: Comparing Predictions

```python
# Predictions
y_pred_rf = rf_model.predict(X)

print(f"Accuracy: {accuracy_score(y, y_pred_rf)}")
print("\\nClassification Report:")
print(classification_report(y, y_pred_rf))

```

**Output Explanation:**

- Random forests often outperform individual decision trees by reducing overfitting.

---

### Code Example 3: Visualizing Tree Diversity

```python
# Visualizing one tree in the forest
plt.figure(figsize=(12, 8))
plot_tree(rf_model.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("A Single Tree in the Random Forest")
plt.show()

```

**Output Explanation:**

- The diversity of trees in the forest helps improve generalization.

---

### Code Example 4: Adjusting Parameters

```python
# Training with limited features per tree
rf_model_limited = RandomForestClassifier(n_estimators=100, max_features=2, random_state=42)
rf_model_limited.fit(X, y)

print("Feature Importances (Limited Features):")
print(dict(zip(iris.feature_names, rf_model_limited.feature_importances_)))

```

**Output Explanation:**

- Limiting features per tree adds randomness and can improve generalization.