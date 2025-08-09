# Module 22: Model Interpretability and Explainability

## What is Model Interpretability and Explainability?

**Model interpretability** refers to the degree to which a human can understand the decisions made by a machine learning model. **Explainability**, on the other hand, is the process of explaining the inner workings of a model, such as how inputs are mapped to outputs.

In machine learning, especially when working with complex models like deep learning and ensemble models, it’s important to understand why a model makes certain predictions. This is where tools like **SHAP** and **LIME** come in—offering methods to explain individual predictions and model behavior.

---

## Section 22.1: SHAP Values

### What is SHAP?

**SHAP** (SHapley Additive exPlanations) is a game theory-based approach to explain individual predictions of machine learning models. It assigns an importance value (SHAP value) to each feature, explaining how much that feature contributes to a particular prediction.

SHAP values have several key advantages:

- They provide **global** and **local** explanations for models.
- SHAP values are based on **Shapley values** from cooperative game theory, ensuring fairness in feature contribution calculations.

---

### Code Example 1: Using SHAP to Explain Model Predictions

**Importing Libraries**

```python
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

```

**Loading Dataset**

```python
# Load a simple dataset (Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

```

**Training a Random Forest Model**

```python
# Train a Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

```

**Explaining the Model with SHAP**

```python
# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Plot the SHAP values for the first prediction
shap.initjs()
shap.force_plot(shap_values[0][0], X_test.iloc[0])

```

**Explanation**

- **TreeExplainer** is used for tree-based models like RandomForest. It computes SHAP values efficiently for each feature and each instance.
- The `force_plot` shows how each feature contributes to the prediction for the first test instance. Positive and negative SHAP values indicate how much each feature pushes the prediction above or below the baseline (expected value).

---

### Output for the Example

**Output**

You will see a force plot displaying the contribution of each feature to the prediction of a specific instance. For example, if a flower's sepal length significantly pushes the prediction towards a specific class, it will be shown as a large positive SHAP value.

---

## Section 22.2: LIME (Local Interpretable Model-agnostic Explanations)

### What is LIME?

**LIME** (Local Interpretable Model-agnostic Explanations) is another technique for explaining individual predictions, but unlike SHAP, it works by approximating the model locally around a given prediction. LIME builds an interpretable surrogate model (e.g., linear regression) that approximates the complex model’s behavior for that particular instance.

LIME is especially useful when dealing with **black-box models** and when you need explanations on specific individual predictions rather than global model behavior.

---

### Code Example 2: Using LIME to Explain Predictions

**Importing Libraries**

```python
import lime
import lime.lime_tabular
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

```

**Loading Dataset and Training a Model**

```python
# Load the dataset and train a Random Forest model
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

```

**Explaining a Prediction with LIME**

```python
# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    mode='classification',
    training_labels=y_train.values,
    feature_names=X_train.columns,
    class_names=data.target_names
)

# Explain the first instance in the test set
explanation = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=4)

# Show the explanation
explanation.show_in_notebook()

```

**Explanation**

- LIME works by creating a locally interpretable surrogate model around the instance being explained. Here, we use a Random Forest model and explain the prediction for the first test instance.
- The `explain_instance` method provides an explanation in terms of feature importance for that particular prediction.

---

### Output for the Example

**Output**

The output will be a visual explanation of the model’s decision for the first test instance. It will show how the model’s prediction is influenced by the features in a way that can be easily understood.

---

## Section 22.3: wandb Library

### What is wandb?

**Weights & Biases (wandb)** is a popular library for tracking experiments in machine learning. It provides tools for logging, visualizing, and comparing models, metrics, and hyperparameters. wandb also allows you to track the performance of your models and generate detailed reports.

wandb is often used for:

- Logging training and validation metrics.
- Visualizing model performance over time.
- Storing and sharing datasets and models.
- Collaborating with teammates.

---

### Code Example 3: Using wandb for Experiment Tracking

**Importing Libraries**

```python
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

```

**Setting Up wandb and Logging Metrics**

```python
# Initialize wandb
wandb.init(project='model-experiment', entity='your_username')

# Load dataset and split
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Log metrics using wandb
wandb.log({'accuracy': model.score(X_test, y_test)})

```

**Explanation**

- The `wandb.init()` function initializes a new experiment in your project. Every time a model is trained, you can log various metrics (like accuracy, loss, or any custom metric).
- The `wandb.log()` function logs metrics to your wandb dashboard, where you can track performance over multiple experiments.

---

### Output for the Example

**Output**

When you run this code, a new experiment will be logged in your **wandb** dashboard, showing the accuracy metric of your trained model. You can compare this experiment with others and visualize the results.