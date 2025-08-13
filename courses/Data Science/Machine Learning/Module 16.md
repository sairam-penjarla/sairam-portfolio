# Module 16: Natural Language Processing (NLP) - Sentiment Analysis

## What is Sentiment Analysis?

**Sentiment Analysis** is a technique used in **Natural Language Processing (NLP)** to identify and extract subjective information from text. The goal is to determine the sentiment expressed by a piece of text, which could be positive, negative, or neutral. It is widely used in various applications, such as:

- **Customer feedback analysis**
- **Social media monitoring**
- **Brand sentiment analysis**
- **Market research**

Sentiment analysis can help businesses and organizations understand how people feel about their products, services, or brands by analyzing text from online reviews, tweets, or forums.

### Types of Sentiment Analysis:

- **Binary Sentiment Classification**: Classifying text into two categories: positive or negative.
- **Fine-grained Sentiment Classification**: Classifying text into multiple categories (e.g., positive, neutral, and negative, or on a scale from 1 to 5).
- **Aspect-based Sentiment Analysis**: Identifying sentiments towards specific aspects or features of a product or service.

---

## Section 16.1: Sentiment Analysis using TextBlob

### TextBlob Overview

**TextBlob** is a simple Python library for processing textual data. It offers an easy-to-use API for common NLP tasks, including **sentiment analysis**. TextBlob uses a lexicon-based approach to determine sentiment by analyzing text and assigning polarity and subjectivity scores.

- **Polarity**: A score between -1 (negative) and 1 (positive), indicating the sentiment's positivity or negativity.
- **Subjectivity**: A score between 0 (objective) and 1 (subjective), indicating the degree to which the text is opinionated.

### Code Example 1: Sentiment Analysis with TextBlob

**Importing Libraries**

```python
from textblob import TextBlob

```

**Explanation**

- We import **TextBlob**, which provides the `TextBlob()` class for analyzing and processing text.

---

**Performing Sentiment Analysis on Text**

```python
# Example text
text = "I love the new features of this product, it's amazing!"

# Create a TextBlob object
blob = TextBlob(text)

# Get the sentiment of the text
sentiment = blob.sentiment

# Print sentiment details
print(f"Polarity: {sentiment.polarity}")
print(f"Subjectivity: {sentiment.subjectivity}")

```

**Explanation**

- We create a **TextBlob** object using the example text.
- The `sentiment` property of the TextBlob object provides a tuple with two values: polarity and subjectivity.
- We print out both values to understand the sentiment of the text.

---

### Output for the Example Text

**Output**

```
Polarity: 0.75
Subjectivity: 0.6

```

**Explanation**

- **Polarity (0.75)** indicates a positive sentiment, as the score is closer to 1.
- **Subjectivity (0.6)** indicates that the text is somewhat subjective, meaning the text contains opinions or personal feelings.

---

## Section 16.2: Sentiment Analysis using VADER

### VADER Overview

**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a sentiment analysis tool specifically designed for social media text, such as tweets. It is lexicon-based, using a list of words and their corresponding sentiment scores, and it accounts for punctuation, capitalization, and emoticons, making it suitable for informal text.

VADER provides a sentiment score that ranges from -1 (negative) to +1 (positive), as well as a **compound** score that represents the overall sentiment.

### Code Example 2: Sentiment Analysis with VADER

**Importing Libraries**

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

```

**Explanation**

- We import **SentimentIntensityAnalyzer** from **VADER** to analyze the sentiment of the text.

---

**Performing Sentiment Analysis with VADER**

```python
# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Example text
text = "I hate the new update, it's really frustrating!"

# Get the sentiment scores
scores = analyzer.polarity_scores(text)

# Print sentiment scores
print(f"Sentiment Scores: {scores}")

```

**Explanation**

- We initialize the **SentimentIntensityAnalyzer**.
- We analyze the sentiment of the given text using the `polarity_scores()` method.
- The `polarity_scores()` method returns a dictionary with sentiment scores, including the **compound score**, which is the overall sentiment.

---

### Output for the Example Text

**Output**

```
Sentiment Scores: {'neg': 0.537, 'neu': 0.463, 'pos': 0.0, 'compound': -0.759}

```

**Explanation**

- **Negative score (0.537)** indicates that the text expresses negativity.
- **Neutral score (0.463)** shows that there’s a neutral tone present as well.
- **Positive score (0.0)** shows no positivity in the text.
- **Compound score (-0.759)**, being negative, indicates an overall negative sentiment.

---

## Section 16.3: Sentiment Analysis using Hugging Face's Transformers

### Hugging Face Transformers Overview

**Hugging Face's Transformers** library offers state-of-the-art NLP models, including transformer-based models such as **BERT**, **RoBERTa**, and **DistilBERT**. These models are pre-trained on large datasets and can be fine-tuned for specific tasks like sentiment analysis. Hugging Face provides an easy-to-use API to load models and perform sentiment analysis.

### Code Example 3: Sentiment Analysis with Hugging Face

**Importing Libraries**

```python
from transformers import pipeline

```

**Explanation**

- We import **pipeline** from the Hugging Face **transformers** library, which allows easy access to pre-trained models for various tasks, including sentiment analysis.

---

**Performing Sentiment Analysis with Hugging Face**

```python
# Load the sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example text
text = "The movie was absolutely fantastic! I would watch it again."

# Get sentiment analysis result
result = sentiment_pipeline(text)

# Print result
print(result)

```

**Explanation**

- We use the **pipeline()** function to load the pre-trained sentiment-analysis model.
- The **pipeline** handles all the heavy lifting, including tokenization, model inference, and result generation.
- The sentiment analysis result contains the label (positive or negative) and the confidence score.

---

### Output for the Example Text

**Output**

```
[{'label': 'POSITIVE', 'score': 0.9998735189437866}]

```

**Explanation**

- The model predicts **POSITIVE** sentiment with a high confidence score (**0.9998**). This is a clear indication that the sentiment of the text is highly positive.

---

## Section 16.4: Custom Sentiment Analysis with Deep Learning Models

### Custom Deep Learning Models for Sentiment Analysis

If you need a more tailored approach or have a specialized dataset, you can train your own deep learning models for sentiment analysis. Models such as **LSTM** (Long Short-Term Memory) or **GRU** (Gated Recurrent Units) can be used for sequence-based tasks like sentiment analysis. These models can learn contextual information and are particularly useful for handling longer sentences and more complex sentiment expressions.

---

**Training a Custom Sentiment Analysis Model using LSTM**

Here’s a high-level outline for training a custom **LSTM-based** sentiment analysis model:

1. **Prepare the dataset**: Collect labeled data with text and sentiment labels (positive, negative, neutral).
2. **Tokenize the text**: Convert text into numerical sequences using tokenization.
3. **Build the model**: Use a deep learning framework like TensorFlow or Keras to build an LSTM model.
4. **Train the model**: Train the model on the prepared dataset.
5. **Evaluate the model**: Measure performance using evaluation metrics like accuracy or F1-score.

This process involves more advanced knowledge of machine learning frameworks and model training, but it's a powerful way to achieve custom sentiment analysis solutions.