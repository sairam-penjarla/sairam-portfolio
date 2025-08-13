# Module 14: Natural Language Processing (NLP) - Word Embeddings (e.g., Word2Vec, GloVe)

## What is Word Embedding?

**Word Embeddings** are a type of **word representation** that allows words with similar meaning to have a similar representation. Unlike traditional methods that represent words as unique, high-dimensional, sparse vectors (e.g., one-hot encoding), word embeddings are dense vectors that capture semantic relationships between words. These vectors are learned from large text corpora and map words to real-valued vectors of fixed dimensions.

The core idea behind word embeddings is that words that are used in similar contexts tend to have similar meanings. By representing words as dense vectors, word embeddings capture these semantic relationships, allowing machines to better understand the meanings behind words and perform tasks such as **text classification**, **sentiment analysis**, and **machine translation**.

### Popular Word Embedding Models:

1. **Word2Vec (Word to Vec)**:
Word2Vec is a neural network-based model introduced by Google that learns word representations by training on large text data. It comes with two training architectures:
    - **Continuous Bag of Words (CBOW)**: Predicts a target word based on context words.
    - **Skip-gram**: Predicts context words based on a target word.
2. **GloVe (Global Vectors for Word Representation)**:
GloVe is a count-based method for learning word embeddings. Unlike Word2Vec, GloVe uses the global statistical information of the corpus (i.e., the co-occurrence matrix) to compute embeddings. It factorizes the word co-occurrence matrix into lower-dimensional vectors, preserving the statistical relationships between words.

---

## Section 14.1: Word2Vec

### Word2Vec Overview

**Word2Vec** transforms words into dense vectors by leveraging the context in which words appear. It uses a shallow neural network model trained on a large corpus of text, and the resulting vectors capture both syntactic and semantic properties of words. Word2Vec has been popular due to its simplicity and efficiency in capturing relationships between words.

### Code Example 1: Training Word2Vec Model with Gensim

**Importing Libraries**

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

```

**Explanation**

- We use **Gensim**, a Python library that specializes in unsupervised learning tasks like topic modeling and vectorization. Here, we are importing the **Word2Vec** class from **gensim.models** to train our model.
- We also import **word_tokenize** from NLTK for text tokenization.

---

**Training a Word2Vec Model**

```python
# Example corpus
corpus = [
    "Machine learning is a method of data analysis.",
    "Word2Vec helps in generating word embeddings.",
    "Gensim provides an easy way to train Word2Vec models."
]

# Tokenize the text
tokenized_corpus = [word_tokenize(text.lower()) for text in corpus]

# Train a Word2Vec model
model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector representation for a word
word_vector = model.wv['word2vec']

print("Vector for 'word2vec':", word_vector)

```

**Explanation**

- We prepare a small **corpus** of sentences.
- We **tokenize** each sentence into words using **word_tokenize** and convert everything to lowercase.
- The **Word2Vec()** function from **Gensim** is used to train the model. We specify:
    - **vector_size=100**: The size of the word vectors.
    - **window=5**: The maximum distance between the current and predicted word within a sentence.
    - **min_count=1**: Ignore words with fewer than 1 occurrence.
    - **workers=4**: Use 4 CPU cores for training.
- We then retrieve the word vector for the word **'word2vec'** using `model.wv['word2vec']`. The output will be a dense vector representing the word.

---

### Word2Vec Example Output

**Output**

The output will be a **dense vector** representing the word **'word2vec'**, such as:

```
Vector for 'word2vec': [ 0.02950498  0.05623491  0.02188921 ...] (100-dimensional vector)

```

This dense vector encodes the meaning of the word **'word2vec'** relative to the other words in the corpus.

---

## Section 14.2: GloVe

### GloVe Overview

**GloVe (Global Vectors for Word Representation)** is an unsupervised learning algorithm for word embeddings. Unlike Word2Vec, which learns word embeddings by predicting context words, GloVe uses a matrix factorization approach. It leverages the global statistical information of a corpus, specifically the co-occurrence matrix, which counts how often words appear together in the text.

The key idea is that the ratio of the co-occurrence probabilities of two words should be preserved in the embedding space.

### Code Example 2: Using Pre-trained GloVe Embeddings

To use GloVe embeddings, we typically download pre-trained word vectors. Here’s an example of how to load and use pre-trained GloVe vectors.

**Importing Libraries**

```python
import numpy as np

```

**Explanation**

- **Numpy** is used to handle numerical arrays, such as word vectors.

---

**Loading Pre-trained GloVe Embeddings**

```python
# Define the GloVe file path (assuming GloVe embeddings are saved as a .txt file)
glove_file = "glove.6B.100d.txt"

# Load the GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load GloVe embeddings
embeddings = load_glove_embeddings(glove_file)

# Retrieve the vector for the word 'king'
king_vector = embeddings['king']

print("Vector for 'king':", king_vector)

```

**Explanation**

- We load pre-trained GloVe embeddings from a file (in this case, the 100-dimensional GloVe embeddings file `glove.6B.100d.txt`).
- For each line in the file, we extract the word and its corresponding vector and store it in the `embeddings` dictionary.
- We retrieve the word vector for **'king'** and print it.

---

### GloVe Example Output

**Output**

The output will be a 100-dimensional vector representing the word **'king'**, such as:

```
Vector for 'king': [ 0.22825    -0.12223    0.47616 ... ] (100-dimensional vector)

```

---

## Section 14.3: Word Embedding Applications

### Applications of Word Embeddings

1. **Semantic Similarity**: Word embeddings allow us to measure the similarity between words. Words with similar meanings, such as "dog" and "puppy," will have similar embeddings, and we can compute the cosine similarity between them to quantify how similar they are.
2. **Word Analogies**: Word embeddings can also solve word analogies like "King - Man + Woman = Queen." This is possible because word embeddings capture relational information.
3. **Text Classification and Sentiment Analysis**: Word embeddings are often used as features in text classification models, such as predicting the sentiment of a review or categorizing news articles.

---

Word embeddings like **Word2Vec** and **GloVe** provide a powerful way to represent words in a dense vector space that captures their meanings and relationships. These embeddings are essential for modern NLP tasks and enable machines to understand and process human language in a more sophisticated manner.