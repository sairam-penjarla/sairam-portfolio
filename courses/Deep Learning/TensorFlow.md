# Table of contents

# Module 1: Neural Networks

## Section 1.1: Perceptron

### What is a Perceptron?

A **Perceptron** is the simplest type of neural network. It consists of a single layer of neurons, where each neuron performs a weighted sum of its inputs, applies an activation function (like step or sigmoid), and produces an output. While it can only handle linear separable problems, it is the foundation of more complex neural networks.

---

### Code Example 1: Implementing a Perceptron in TensorFlow

**Importing Libraries**

```python
import tensorflow as tf
import numpy as np

```

**Dataset Creation**

```python
# Creating a simple binary classification dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # Inputs
y = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Outputs (AND logic)

```

**Building the Perceptron Model**

```python
# Define a simple Perceptron model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
])

```

**Compiling the Model**

```python
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

```

**Training the Model**

```python
# Train the model
model.fit(X, y, epochs=100, verbose=0)

```

**Evaluating and Predicting**

```python
# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(X)
print("Predictions:")
print(predictions)

```

---

### Step-by-Step Explanation of the Code

1. **Dataset Creation**: We define the inputs (`X`) and outputs (`y`) of an AND logic gate. The perceptron learns to map these inputs to the correct output.
2. **Model Definition**: The `Dense` layer represents the perceptron. `units=1` means there's a single neuron, and the `sigmoid` activation function squashes the output to a range between 0 and 1.
3. **Compilation**: We use the **Adam optimizer** for training and **binary_crossentropy** as the loss function for binary classification tasks.
4. **Training**: The model is trained for 100 epochs on the dataset. Verbose mode is disabled for cleaner output.
5. **Evaluation and Prediction**: The model is evaluated for accuracy, and predictions are made for all inputs. Outputs close to 1 indicate a `True` prediction.

---

## Section 1.2: Feedforward Neural Networks (FNN)

### What is a Feedforward Neural Network?

A **Feedforward Neural Network (FNN)** is a multi-layer perceptron where data flows only in one direction—from input to output. Unlike a perceptron, it can handle complex, non-linear problems by using multiple layers and non-linear activation functions.

---

### Code Example 2: Implementing an FNN in TensorFlow

**Dataset Preparation**

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate a dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # Reshape target to be compatible with TensorFlow

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

**Defining the Model**

```python
# Define an FNN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

```

**Compiling the Model**

```python
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

```

**Training the Model**

```python
# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, verbose=0)

```

**Evaluating the Model**

```python
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

```

---

### Step-by-Step Explanation of the Code

1. **Dataset**: We use `make_moons`, a non-linear dataset, to show the power of FNNs. The dataset is split into training and testing sets.
2. **Model Definition**: The network has:
    - **Layer 1**: 16 neurons with ReLU activation.
    - **Layer 2**: 8 neurons with ReLU activation.
    - **Output Layer**: A single neuron with sigmoid activation for binary classification.
3. **Compilation**: The same optimizer and loss function as before are used.
4. **Training**: The model is trained for 50 epochs, with 20% of the training data used for validation.
5. **Evaluation**: The model is tested on unseen data, and accuracy is reported.

---

## Section 1.3: Backpropagation and Gradient Descent

### What is Backpropagation?

**Backpropagation** is the algorithm used to train neural networks. It calculates the error (loss) at the output layer and propagates it back through the network to adjust the weights and biases using **gradient descent**.

---

### Code Example 3: Understanding Backpropagation in TensorFlow

**Dataset**

```python
# Using the same dataset as in Section 1.2

```

**Defining the Model**

```python
# Define the same model as before
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

```

**Custom Training Loop with Backpropagation**

```python
# Custom training loop
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Training for 100 epochs
for epoch in range(100):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(X_train)
        loss = loss_fn(y_train, predictions)

    # Backward pass
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Logging
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

```

---

### Step-by-Step Explanation of the Code

1. **Custom Training Loop**:
    - **GradientTape**: Tracks operations for automatic differentiation.
    - **Loss Calculation**: Computes the binary cross-entropy loss between predictions and true labels.
    - **Gradients**: Calculates gradients of the loss with respect to trainable variables (weights and biases).
    - **Optimizer**: Updates the weights using gradient descent via the Adam optimizer.
2. **Logging**: The loss is logged every 10 epochs for visibility into training progress.

---

# Module 2: Training Deep Neural Networks

## Section 2.1: Loss Functions

### What is a Loss Function?

A **loss function** measures the difference between the predicted output of a neural network and the true target values. It guides the optimization process by quantifying the error the model needs to minimize. The choice of loss function depends on the task:

1. **Mean Squared Error (MSE)**: Commonly used for regression problems.
2. **Binary Cross-Entropy**: Used for binary classification tasks.
3. **Categorical Cross-Entropy**: Used for multi-class classification tasks.

---

### Code Example 1: Using Different Loss Functions in TensorFlow

**Importing Libraries**

```python
import tensorflow as tf
import numpy as np

```

**Dataset**

```python
# Generate a simple dataset
X = np.random.rand(100, 1).astype(np.float32) * 10  # Random numbers between 0 and 10
y = 3 * X + 5 + np.random.normal(0, 2, (100, 1))  # y = 3x + 5 + noise

```

**Model Definition**

```python
# Define a simple regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,))
])

```

**Using Mean Squared Error**

```python
# Compile the model with Mean Squared Error loss
model.compile(optimizer='adam', loss='mse')

```

**Training the Model**

```python
# Train the model
history = model.fit(X, y, epochs=50, verbose=0)

```

---

### Step-by-Step Explanation

1. **Dataset**: A regression dataset is created using a linear function with added noise.
2. **Model Definition**: A single dense layer maps inputs to outputs.
3. **Loss Function**: The Mean Squared Error (MSE) loss function is specified, which calculates the average squared difference between predicted and actual values.
4. **Training**: The model learns to minimize the loss by adjusting weights and biases.

---

## Section 2.2: Optimizers

### What is an Optimizer?

An **optimizer** updates the weights and biases of a neural network to minimize the loss function. Different optimizers control how the weights are updated during training. Common optimizers include:

1. **SGD (Stochastic Gradient Descent)**: Updates weights using gradients from a random subset (batch) of data.
2. **Adam (Adaptive Moment Estimation)**: Combines the benefits of SGD and RMSProp, offering adaptive learning rates.

---

### Code Example 2: Comparing Optimizers

**Using Stochastic Gradient Descent**

```python
# Compile the model with SGD optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')
model.fit(X, y, epochs=50, verbose=0)

```

**Using Adam**

```python
# Compile the model with Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
model.fit(X, y, epochs=50, verbose=0)

```

**Evaluating the Performance**

```python
# Make predictions
predictions = model.predict(X)
print(predictions[:5])  # Display first 5 predictions

```

---

### Step-by-Step Explanation

1. **SGD**: Optimizes the model by iteratively updating weights using a batch of data.
2. **Adam**: Adjusts learning rates during training, improving convergence for complex problems.
3. **Comparison**: Both optimizers minimize the same loss but may do so at different speeds or stabilities.

---

## Section 2.3: Regularization Techniques

### Why Use Regularization?

**Regularization** prevents overfitting by reducing the model's capacity to memorize the training data. This is especially crucial for deep neural networks. Common techniques include:

1. **Dropout**: Randomly sets a fraction of activations to zero during training.
2. **Batch Normalization**: Normalizes activations across a batch to stabilize and speed up training.

---

### Code Example 3: Applying Dropout and Batch Normalization

**Defining the Model with Dropout**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

```

**Adding Batch Normalization**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

```

**Training the Model**

```python
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)

```

---

### Step-by-Step Explanation

1. **Dropout**: Regularizes the model by randomly deactivating neurons during training. The `Dropout` layer takes a probability (e.g., `0.5`).
2. **Batch Normalization**: Normalizes layer inputs to have zero mean and unit variance, improving training stability.
3. **Model Definition**: The techniques are integrated seamlessly within the model.

---

This module lays a strong foundation for understanding how to train deep neural networks effectively using loss functions, optimizers, and regularization techniques.