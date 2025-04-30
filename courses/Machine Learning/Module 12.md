# Module 12: Cross-validation and Model Evaluation Metrics

## What is Cross-validation?

**Cross-validation** is a technique used to evaluate the performance of a machine learning model by splitting the dataset into multiple subsets or **folds**. The model is trained on some of these folds and tested on the remaining ones. This process is repeated several times to get a more accurate estimate of how the model will perform on unseen data.

One of the most common cross-validation techniques is **k-fold cross-validation**, where the data is divided into **k** subsets. The model is trained **k-1** times, each time using a different subset as the test set, and the performance metrics are averaged.

The key benefits of cross-validation are:

- It provides a more reliable estimate of model performance.
- It reduces the risk of overfitting by evaluating the model on different data splits.

---

## Section 12.1: K-Fold Cross-Validation

### How K-Fold Cross-Validation Works:

In **k-fold cross-validation**, the dataset is split into **k** folds (subsets). For each fold, the model is trained on **k-1** folds and tested on the remaining fold. This process is repeated **k** times, each time with a different fold as the test set. The final model performance is averaged across all **k** iterations.

This technique helps in assessing how well the model generalizes to different subsets of the data, making it a valuable tool for model evaluation.

### Code Example 1: K-Fold Cross-Validation using Scikit-learn

**Importing Libraries and Dataset**

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

```

**Explanation**

- We use the **Iris dataset**, a well-known dataset for classification tasks, and **Logistic Regression** as the model.
- The **cross_val_score** function is used to perform k-fold cross-validation and return the accuracy of the model for each fold.

---

**Running K-Fold Cross-Validation**

```python
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the model
model = LogisticRegression(max_iter=200)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())

```

**Explanation**

- The **cross_val_score** function splits the dataset into 5 folds (cv=5) and evaluates the model's performance on each fold.
- The scores are printed for each fold, and the **average accuracy** is computed.

---

## Section 12.2: Model Evaluation Metrics

### What Are Model Evaluation Metrics?

**Model evaluation metrics** are used to assess the performance of machine learning models. They help determine how well a model is making predictions and provide insights into the model's strengths and weaknesses.

Common evaluation metrics include:

- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall (Sensitivity)**: The proportion of actual positive instances that are correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC Curve and AUC**: Measures of the model's ability to distinguish between classes.

The choice of metric depends on the nature of the problem and the importance of different types of errors (false positives, false negatives).

---

## Section 12.3: Accuracy, Precision, Recall, and F1-Score

### Code Example 2: Evaluating Classification Model Performance

**Importing Libraries**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

```

**Explanation**

- We use **accuracy_score**, **precision_score**, **recall_score**, and **f1_score** from Scikit-learn to calculate these metrics.

---

**Splitting Dataset and Training a Model**

```python
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

```

**Explanation**

- We split the dataset into training and test sets (70% train, 30% test).
- A **Logistic Regression** model is trained on the training data and used to make predictions on the test data.

---

**Calculating Performance Metrics**

```python
# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

```

**Explanation**

- We calculate the **accuracy**, **precision**, **recall**, and **F1-score** for the model's performance on the test set.
- The `average='weighted'` option is used to account for class imbalance by weighting metrics by the number of samples in each class.

---

## Section 12.4: ROC Curve and AUC

### What is the ROC Curve and AUC?

The **Receiver Operating Characteristic (ROC)** curve is a graphical representation of a model's ability to distinguish between the positive and negative classes. It plots the **True Positive Rate (Recall)** against the **False Positive Rate (1 - Specificity)** for various threshold values.

The **AUC (Area Under the Curve)** represents the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance. An AUC score of 1 indicates perfect classification, while a score of 0.5 indicates random guessing.

### Code Example 3: ROC Curve and AUC

**Importing Libraries**

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

```

**Explanation**

- We use **roc_curve** to calculate the false positive rate and true positive rate at various thresholds, and **roc_auc_score** to compute the AUC.

---

**Plotting the ROC Curve**

```python
# Get the probability scores for the test set
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

```

**Explanation**

- The **predict_proba** method is used to get the probability scores for each class (not just the predicted class). The second column (`[:, 1]`) contains the probabilities for the positive class.
- We plot the ROC curve and display the AUC score on the plot.

---

Cross-validation and model evaluation metrics are essential for ensuring your model generalizes well and performs effectively on unseen data. By using these techniques, you can assess model performance comprehensively and make informed decisions about model improvements.