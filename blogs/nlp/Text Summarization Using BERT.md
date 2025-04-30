<!--
title: Text Summarization Using BERT
date: July 17, 2024
thumbnail: /static/images/machine-learning/cust-seg.png
-->

# Text Summarization Using BERT: A Comprehensive Guide for Intermediate Practitioners  
🗓️ *June 27, 2024*

Text summarization is a crucial Natural Language Processing (NLP) task that helps condense large volumes of text into concise, meaningful insights. In this guide, we explore how to use BERT, a transformer-based model, to build a custom text summarizer. From understanding the theory to breaking down the implementation block by block, this post offers everything you need to create your own summarization pipeline.

---

### 🔗 GitHub Repository  
You can find the full project code, including `requirements.txt` and dataset files, on GitHub:  
👉 [https://github.com/sairam-penjarla/NLP-Text-Summariser-BERT](https://github.com/sairam-penjarla/NLP-Text-Summariser-BERT)

---

## 🧠 Understanding Text Summarization

### What is Text Summarization?  
Text summarization involves reducing a large body of text into a shorter version that preserves its key meaning. It is especially useful in domains such as research, journalism, and business intelligence where digesting large documents quickly is essential.

### Why BERT for Summarization?  
BERT (Bidirectional Encoder Representations from Transformers) captures deep contextual relationships by looking at words in both directions (left-to-right and right-to-left). Its attention mechanism and contextual understanding make it a strong foundation for extractive summarization.

---

## 🧰 Step-by-Step Implementation

### 1. **Importing the Required Libraries**
```python
import pandas as pd
from summarizer import Summarizer
```
We import `pandas` for data handling and the `Summarizer` module for BERT-based text summarization.

---

### 2. **Creating the `TextSummarizer` Class**
```python
class TextSummarizer:
    def __init__(self, max_length=400, min_length_text=40):
        self.MAX_LENGTH = max_length
        self.MIN_LENGTH_TEXT = min_length_text
        self.summarizer = Summarizer()
```
This class encapsulates the summarization logic. You can tweak the `max_length` and `min_length_text` parameters to control the size of the input blocks and summary outputs.

---

### 3. **Summarizing a Text Block**
```python
def summarize_text(self, text):
    summary = self.summarizer(text, min_length=self.MIN_LENGTH_TEXT)
    return ''.join(summary)
```
This method takes raw text and returns a BERT-generated summary. It's the core of our summarization engine.

---

### 4. **Reading and Processing Dataset**
```python
def process_data(self, data_path, num_blocks=5):
    DATA_COLUMNS = {
        'TEXT': str, 'ENV_PROBLEMS': int, 'POLLUTION': int,
        'TREATMENT': int, 'CLIMATE': int, 'BIOMONITORING': int
    }
    df = pd.read_csv(data_path, delimiter=';', header=0)
    df = df.astype(DATA_COLUMNS)
    df = df[:num_blocks]

    bodies, i = [], 0
    while i < len(df):
        body, body_empty = "", True
        while len(body) < self.MAX_LENGTH and i < len(df):
            body += df.loc[i, 'TEXT'] if body_empty else " " + df.loc[i, 'TEXT']
            body_empty = False
            i += 1
        bodies.append(body)

    for body in bodies:
        print("ORIGINAL TEXT:\n", body)
        print("\nBERT Summary:\n", self.summarize_text(body))
```
This function loads and preprocesses the CSV data, forms blocks of text, and summarizes each using the previously defined method. It's a great way to work with multi-line or document-level data.

---

### 5. **Using the Summarizer in Practice**
```python
text_summarizer = TextSummarizer()
text_summarizer.process_data('water_problem_nlp_en_for_Kaggle_100.csv')

# Manual summary example
body = "Despite the similar volumes of discharged wastewater major part of pollutants comes with communal WWTPs..."
summary = text_summarizer.summarizer(body)
print(summary)
```
This block demonstrates how to instantiate the `TextSummarizer` class and use it both for processing a dataset and summarizing standalone text.

---

## ✅ Conclusion

In this blog, we explored how to build a custom text summarization tool using BERT. You’ve learned:

- The theory behind text summarization and BERT
- How to preprocess and chunk textual data
- How to implement summarization using Python
- How to evaluate and apply the summarizer in real-world scenarios

By following this guide and exploring the [GitHub repo](https://github.com/sairam-penjarla/NLP-Text-Summariser-BERT), you'll have a solid foundation to apply BERT to any text summarization task—be it academic papers, news articles, or survey responses.

Happy summarizing! 🚀
