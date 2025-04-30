# Implementing a Random Forest Classifier from Scratch in Python
**📅 Date:** July 17, 2024**

In this blog post, we will walk through the process of implementing a **Random Forest classifier** from scratch in Python. The Random Forest algorithm is a powerful machine learning technique that combines multiple decision trees to enhance classification accuracy while reducing the risk of overfitting. By the end of this guide, you will gain a comprehensive understanding of how the components of a Random Forest work together to generate accurate predictions.

To get hands-on with the implementation, you can clone the repository and follow along. This practical approach will help solidify your understanding of the concepts discussed. Let's dive in!

### Cloning the Repository

Start by cloning the repository using the following commands:

```bash
# Clone the repository
git clone https://github.com/sairam-penjarla/Random-Forest-from-scratch.git

# Navigate to the project directory
cd Random-Forest-from-scratch
```

### Understanding the Code Structure

The project consists of four main files:

- `tree.py`
- `decision_tree.py`
- `random_forest.py`
- `inference.py`

Each file plays a crucial role in building the Random Forest classifier. Let's explore them in detail.

---

### `tree.py` - Core Classes and Functions

This file contains the `Node` class and the `resample_data` function, both of which are vital for constructing decision trees and generating bootstrap samples.

#### **resample_data**

The `resample_data` function creates a bootstrap sample from the dataset by randomly sampling data points with replacement to create a new dataset of the same size as the original.

```python
import numpy as np

def resample_data(features, labels):
    """Generate a bootstrap sample from the dataset."""
    num_samples = features.shape[0]
    sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)
    return features[sample_indices], labels[sample_indices]
```

#### **Node Class**

The `Node` class represents a single node in a decision tree. It contains information such as the depth of the node, the feature and threshold used for splitting, and pointers to the left and right child nodes.

```python
class Node:
    def __init__(self, current_depth=0, depth_limit=None):
        """Initialize a tree node."""
        self.current_depth = current_depth
        self.depth_limit = depth_limit
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None  # Value for leaf nodes
```

---

### `decision_tree.py` - Building a Decision Tree

The `decision_tree.py` file contains the `CustomDecisionTree` class, which encapsulates the logic for building and using a single decision tree.

#### **CustomDecisionTree Class**

This class is initialized with an optional depth limit, which controls the maximum depth of the tree. 

```python
import numpy as np
from tree import Node, resample_data

class CustomDecisionTree:
    def __init__(self, depth_limit=None):
        """Initialize the CustomDecisionTree with an optional depth limit."""
        self.depth_limit = depth_limit
```

#### **fit Method**

The `fit` method trains the decision tree using the provided features and labels, initializing the tree structure and starting the construction process.

```python
def fit(self, features, labels):
    """Fit the CustomDecisionTree to the provided features and labels."""
    self.num_classes = len(np.unique(labels))
    self.num_features = features.shape[1]
    self.root = self._build_tree(features, labels)
```

#### **_build_tree Method**

This recursive method builds the decision tree by repeatedly finding the best splits based on the Gini impurity until the depth limit or stopping criteria are reached.

```python
def _build_tree(self, features, labels, current_depth=0):
    """Recursively build the tree."""
    class_counts = [np.sum(labels == c) for c in range(self.num_classes)]
    majority_class = np.argmax(class_counts)
    node = Node(current_depth, self.depth_limit)
    # Further splitting logic
    return node
```

#### **_find_optimal_split Method**

The `_find_optimal_split` method identifies the best feature and threshold for splitting the data to minimize the Gini impurity.

```python
def _find_optimal_split(self, features, labels, feature_indices):
    """Find the best feature and threshold for splitting the data."""
    # Split finding logic
    return optimal_feature, optimal_threshold
```

---

### `random_forest.py` - Implementing the Random Forest Classifier

This file contains the `RandomForestClassifier` class, which implements the Random Forest ensemble. It manages the training of multiple decision trees and aggregates their predictions.

#### **RandomForestClassifier Class**

The `RandomForestClassifier` is initialized with the number of trees (`num_trees`) and an optional depth limit for individual decision trees.

```python
import numpy as np
from decision_tree import CustomDecisionTree

class RandomForestClassifier:
    def __init__(self, num_trees=100, depth_limit=None):
        """Initialize the RandomForestClassifier with the number of trees and an optional depth limit."""
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.trees = []
```

#### **fit Method**

The `fit` method trains the Random Forest by creating multiple decision trees (instances of `CustomDecisionTree`) and fitting them to bootstrap samples of the data.

```python
def fit(self, features, labels):
    """Fit the RandomForestClassifier to the provided features and labels."""
    self.trees = []
    for _ in range(self.num_trees):
        tree = CustomDecisionTree(depth_limit=self.depth_limit)
        sample_features, sample_labels = self._create_bootstrap_sample(features, labels)
        tree.fit(sample_features, sample_labels)
        self.trees.append(tree)
```

#### **predict Method**

The `predict` method aggregates predictions from all trees in the forest, providing a final prediction by calculating the mean prediction across all trees.

```python
def predict(self, features):
    """Predict the class labels for the given features."""
    predictions = np.zeros((features.shape[0], self.num_trees))
    for idx, tree in enumerate(self.trees):
        predictions[:, idx] = tree.predict(features)
    return np.mean(predictions, axis=1)
```

---

### `inference.py` - Example Usage

The `inference.py` file demonstrates the practical usage of the `RandomForestClassifier` on the Iris dataset.

#### **Example Workflow with Iris Dataset**

Here’s how you can use the `RandomForestClassifier` to train and evaluate a model on the Iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForestClassifier

# Load the Iris dataset
iris_data = load_iris()
features, labels = iris_data.data, iris_data.target

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
random_forest = RandomForestClassifier(num_trees=100, depth_limit=5)
random_forest.fit(features_train, labels_train)

# Make predictions on the test set
predicted_labels = random_forest.predict(features_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(labels_test, predicted_labels.round())
print(f'Accuracy: {accuracy * 100:.2f}%')
```

This snippet walks through the process of loading the dataset, training the Random Forest model, making predictions, and evaluating the model’s accuracy.

---

### Conclusion

In this blog post, we have explored how to implement a **Random Forest classifier** from scratch using Python. Here's a summary of what we covered:

- **Random Forest Overview**: We introduced the concept of Random Forests, which combine multiple decision trees trained on bootstrap samples of the dataset using random feature subsets.
- **Decision Tree Implementation**: We walked through the creation of a custom decision tree implementation (`CustomDecisionTree`) capable of recursive splitting based on Gini impurity.
- **Random Forest Classifier**: We detailed the implementation of the `RandomForestClassifier` that aggregates the predictions of multiple decision trees to improve performance.
- **Practical Example**: We showed how to apply the Random Forest classifier to the Iris dataset, including training, prediction, and model evaluation.

By understanding how Random Forests work and building them from scratch, you can gain valuable insights into ensemble learning and decision tree models. We encourage you to explore the GitHub repository for further experimentation. Modify parameters, experiment with different datasets, and enhance your understanding of this powerful machine learning technique.

Happy coding!