# Module 17: Natural Language Processing (NLP) - Text Classification

## What is Text Classification?

**Text Classification** is a fundamental task in Natural Language Processing (NLP) where we assign predefined labels or categories to a given text based on its content. It's also known as **Document Classification** or **Text Categorization**. Text classification has a wide range of applications, including:

- **Spam detection** in emails.
- **Sentiment analysis**, as discussed earlier.
- **Topic categorization** for news articles, blogs, etc.
- **Language identification** and more.

Text classification is essential in various domains such as content moderation, customer service, and information retrieval, helping automate and streamline processes that would otherwise require manual sorting.

### Types of Text Classification:

- **Binary Classification**: Classifying text into two categories (e.g., spam vs. not spam).
- **Multiclass Classification**: Classifying text into multiple categories (e.g., categorizing a news article into business, politics, or sports).
- **Multilabel Classification**: Assigning multiple labels to a text (e.g., a product review could be tagged as both "positive" and "helpful").

---

## Section 17.1: Text Classification using Scikit-learn

### Scikit-learn Overview

**Scikit-learn** is a powerful and easy-to-use Python library for machine learning. It provides efficient tools for various classification tasks, including text classification. Scikit-learn uses **Bag-of-Words** or **TF-IDF** (Term Frequency-Inverse Document Frequency) for vectorizing text, which converts textual data into numerical data that machine learning algorithms can process.

### Code Example 1: Text Classification with Scikit-learn

**Importing Libraries**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

```

**Explanation**

- **TfidfVectorizer**: Converts a collection of text documents into a matrix of TF-IDF features.
- **MultinomialNB**: A Naive Bayes classifier used for text classification, especially with categorical data.
- **train_test_split**: Splits the dataset into training and testing sets.
- **accuracy_score**: Measures the accuracy of the classification model.

---

**Preparing the Dataset and Training the Model**

```python
# Sample dataset: Sentences with their respective labels
texts = [
    "This product is great", "Worst purchase I ever made", "Amazing customer service",
    "Do not buy this item", "Very satisfied with the quality", "Terrible quality"
]
labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_tfidf)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

```

**Explanation**

- The **texts** variable contains a list of sample sentences, and **labels** are their corresponding sentiments (positive/negative).
- The dataset is split into **training** and **testing** sets using `train_test_split()`.
- **TF-IDF** is applied to the training and test text data to convert them into a matrix of numerical features.
- We initialize a **Naive Bayes classifier** (`MultinomialNB`) and train it using the vectorized training data.
- We evaluate the model's performance by calculating the **accuracy** using the test data.

---

### Output for the Example Text

**Output**

```
Accuracy: 100.00%

```

**Explanation**

- The model achieved **100% accuracy** on the test data in this simple example, indicating it correctly predicted the sentiment of each sentence.

---

## Section 17.2: Text Classification using Hugging Face's Transformers

### Hugging Face Transformers Overview

**Hugging Face's Transformers** is a state-of-the-art NLP library that provides pre-trained transformer models for various tasks, including text classification. It offers models like **BERT**, **RoBERTa**, **DistilBERT**, and more, which are pre-trained on large corpora and can be fine-tuned for text classification tasks.

### Code Example 2: Text Classification with Hugging Face

**Importing Libraries**

```python
from transformers import pipeline

```

**Explanation**

- We import the **pipeline** function from the **transformers** library, which simplifies using pre-trained models for tasks like text classification.

---

**Performing Text Classification with Hugging Face**

```python
# Load the text classification pipeline
text_classification_pipeline = pipeline("text-classification")

# Example text for classification
text = "I absolutely love this new phone, it's the best I've ever had!"

# Get classification result
result = text_classification_pipeline(text)

# Print the classification result
print(result)

```

**Explanation**

- We use the **pipeline()** function to load the pre-trained **text-classification** model.
- The **pipeline** function automatically handles tokenization, model inference, and output generation.
- We provide an example text for classification and print the result, which shows the predicted label (e.g., positive or negative) with a confidence score.

---

### Output for the Example Text

**Output**

```
[{'label': 'POSITIVE', 'score': 0.999875307559967}]

```

**Explanation**

- The model predicts a **POSITIVE** sentiment with a high confidence score (**0.9999**), showing that the text is classified as positive.

---

## Section 17.3: Text Classification using FastText

### FastText Overview

**FastText** is a library developed by Facebook for efficient learning of word representations and text classification. It is particularly well-suited for working with large datasets and for real-time prediction tasks. FastText uses subword information (e.g., n-grams) to capture more details about the words, making it useful for languages with rich morphology or unknown words.

### Code Example 3: Text Classification with FastText

**Importing Libraries**

```python
import fasttext

```

**Explanation**

- We import **fasttext**, which is used for both word representation and text classification.

---

**Training a Text Classification Model with FastText**

```python
# Sample dataset: Sentences and their labels
train_data = [
    "__label__positive I love this movie",
    "__label__negative This movie was terrible"
]

# Save the dataset to a file
with open("train.txt", "w") as f:
    f.write("\\n".join(train_data))

# Train the FastText model
model = fasttext.train_supervised(input="train.txt")

# Predict the label of a new text
text = "The movie was great!"
prediction = model.predict(text)

# Print prediction result
print(f"Prediction: {prediction[0]} with confidence {prediction[1][0]:.4f}")

```

**Explanation**

- The dataset consists of labeled text where the label is prefixed to the text (e.g., `__label__positive`).
- We save the training data to a file and use **FastText**'s `train_supervised()` function to train a supervised model.
- We then use the model to predict the label of a new sentence, and the model returns both the predicted label and the confidence score.

---

### Output for the Example Text

**Output**

```
Prediction: ('__label__positive',) with confidence 0.9992

```

**Explanation**

- The model predicts the **positive** sentiment for the text with a confidence of **99.92%**.

---

## Section 17.4: Custom Text Classification with Deep Learning

### Custom Deep Learning Models for Text Classification

For more complex or domain-specific tasks, you may need to train a custom deep learning model for text classification. Models like **CNNs** (Convolutional Neural Networks), **LSTMs**, or **Transformers** can be used for text classification tasks, especially when dealing with large datasets or intricate patterns in text.

Here’s a high-level outline for training a custom **LSTM-based** model for text classification:

1. **Prepare the dataset**: Collect a large dataset with labeled text.
2. **Tokenize the text**: Convert text into numerical format using tokenization.
3. **Build the model**: Use deep learning frameworks like **TensorFlow** or **PyTorch** to create an LSTM-based model.
4. **Train the model**: Train the model on the labeled dataset.
5. **Evaluate the model**: Evaluate the model’s performance using metrics such as accuracy or F1-score.

This approach requires more computational resources and expertise but can provide highly accurate results for specific tasks.

---

In this module, we've covered several techniques for **Text Classification**, including:

- **Scikit-learn** with **TF-IDF** and **Naive Bayes**.
- **Hugging Face Transformers**, a cutting-edge approach using pre-trained models.
- **FastText**, a highly efficient library for text classification.
- Custom deep learning approaches using models like **LSTM**.

Each method has its own strengths, and the choice of method depends on the complexity of your task and the available resources.