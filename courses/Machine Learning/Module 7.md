# Module 7: Naive Bayes

## What is Naive Bayes?

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem. It is primarily used for classification tasks and is known for its simplicity and efficiency. The algorithm is termed "naive" because it assumes that all features are independent of each other, which is often not true in real-world data. Despite this assumption, it works surprisingly well for various applications, including text classification, spam detection, and sentiment analysis.

### Bayes' Theorem

Bayes' Theorem provides a mathematical framework for updating probabilities based on new evidence. It is expressed as:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

Where:

- \( P(A|B) \): Posterior probability of \( A \) given \( B \).
- \( P(B|A) \): Likelihood of \( B \) given \( A \).
- \( P(A) \): Prior probability of \( A \).
- \( P(B) \): Probability of \( B \) (normalization factor).

### Naive Bayes Formula for Classification

Given a class \( C \) and features \( X = \{x_1, x_2, ..., x_n\} \), the classifier predicts the class with the highest posterior probability:

\[
C_{\text{pred}} = \arg\max_C P(C) \prod_{i=1}^n P(x_i|C)
\]

Where:

- \( P(C) \) is the prior probability of class \( C \).
- \( P(x_i|C) \) is the likelihood of feature \( x_i \) given class \( C \).

### Types of Naive Bayes Classifiers

1. **Gaussian Naive Bayes**: Assumes continuous features follow a normal distribution.
2. **Multinomial Naive Bayes**: Suitable for discrete data, commonly used in text classification.
3. **Bernoulli Naive Bayes**: Works with binary/boolean data, useful for text-based data where features are word presence/absence.

---

## Code Example 1: Gaussian Naive Bayes on the Iris Dataset

**Importing Libraries and Dataset**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

```

**Explanation:**

- **GaussianNB**: Assumes features follow a Gaussian (normal) distribution.
- The accuracy score and classification report evaluate the model's performance.

---

## Code Example 2: Multinomial Naive Bayes for Text Classification

**Importing Libraries and Dataset**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample dataset
documents = [
    "I love programming",
    "Python is awesome",
    "I hate bugs",
    "Debugging is fun",
    "I love solving problems",
    "Errors are frustrating"
]
labels = [1, 1, 0, 1, 1, 0]  # 1: Positive sentiment, 0: Negative sentiment

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)

print("Predictions:", y_pred)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

```

**Explanation:**

- **CountVectorizer**: Converts text data into a bag-of-words representation.
- **MultinomialNB**: Suited for text classification tasks where features are word counts or frequencies.

---

## Code Example 3: Bernoulli Naive Bayes for Binary Features

**Importing Libraries and Dataset**

```python
from sklearn.naive_bayes import BernoulliNB

# Sample binary dataset
binary_data = [
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 1]
]
labels = [1, 1, 0, 0]  # Binary classification labels

# Train the Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(binary_data, labels)

# Make predictions
test_data = [[1, 0, 0, 0], [0, 1, 1, 1]]
predictions = bnb.predict(test_data)

print("Predictions for Test Data:", predictions)

```

**Explanation:**

- **BernoulliNB**: Assumes binary features, where 1 indicates presence and 0 indicates absence of a feature.

---

## Code Example 4: Visualizing the Decision Boundaries of Naive Bayes

**Visualizing Gaussian Naive Bayes**

```python
import numpy as np
import matplotlib.pyplot as plt

# Using only two features for visualization
X_2d = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

# Train Gaussian Naive Bayes
gnb_2d = GaussianNB()
gnb_2d.fit(X_train, y_train)

# Create a mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict for each point in the grid
Z = gnb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.8, cmap='Pastel1')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='Set1', edgecolor='k')
plt.title("Gaussian Naive Bayes Decision Boundaries")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

```

**Explanation:**

- The contour plot shows how Gaussian Naive Bayes separates the feature space based on probabilities.
- Each region corresponds to the predicted class.