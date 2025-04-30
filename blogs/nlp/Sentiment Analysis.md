# 💬 Sentiment Analysis Using Machine Learning  
**📅 Date:** June 26, 2024  

Explore the power of machine learning and natural language processing (NLP) by building your own sentiment analysis system. In this project, we use real-world tweet data and multiple ML algorithms to classify sentiments as *positive* or *negative*. Whether you're a data scientist or just exploring NLP, this guide walks you through every step — from preprocessing text to deploying models.

---

## 📁 GitHub Repository  

Access the full code, dataset, and `requirements.txt` here:  
🔗 [NLP-Text-Sentiment-Analysis-Bernoulli-Naive-Bayes](https://github.com/sairam-penjarla/NLP-Text-Sentiment-Analysis-Bernoulli-Naive-Bayes)

---

## 🧠 Theory Behind Sentiment Analysis

### What Is Sentiment Analysis?  
Sentiment analysis is the process of identifying the emotional tone behind words. It helps companies track customer feedback, monitor brand reputation, and gauge public opinion — all through the lens of machine learning.

### Why Machine Learning?  
With labeled data, we can train models that recognize patterns in text and classify new data accordingly. This allows automated, scalable sentiment analysis.

---

## 🛠️ Step-by-Step Implementation

### 1️⃣ Importing Required Libraries  
We use Python’s rich ecosystem of libraries for:
- Data manipulation: `pandas`, `numpy`
- Text cleaning: `re`, `WordNetLemmatizer`
- Visualization: `seaborn`, `matplotlib`, `WordCloud`
- Machine learning: `scikit-learn`
- Model persistence: `pickle`

---

### 2️⃣ Loading and Preparing Data

We load a tweet dataset and convert labels:
```python
dataset = pd.read_csv('...csv', encoding="ISO-8859-1", names=columns)
dataset = dataset[['sentiment', 'text']]
dataset['sentiment'] = dataset['sentiment'].replace(4, 1)
```
- Sentiment `0`: Negative  
- Sentiment `1`: Positive

---

### 3️⃣ Text Preprocessing

Clean the tweets using a custom `preprocess()` function:
- Convert to lowercase
- Replace URLs and usernames
- Normalize repeated characters
- Lemmatize words
- Remove non-alphanumerics  

This ensures uniformity for vectorization.

---

### 4️⃣ Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.05)
```

---

### 5️⃣ Feature Extraction: TF-IDF

We use `TfidfVectorizer` to convert text into numeric form:
```python
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
X_train = vectoriser.fit_transform(X_train)
X_test = vectoriser.transform(X_test)
```

---

### 6️⃣ Model Training & Evaluation

We train 3 classifiers:

```python
BNBmodel = BernoulliNB(alpha=2)
SVCmodel = LinearSVC()
LRmodel  = LogisticRegression(C=2, max_iter=1000)
```

Evaluate using a confusion matrix + classification report:
```python
def model_Evaluate(model): ...
```

---

### 7️⃣ Saving Models

```python
pickle.dump(vectoriser, open('vectoriser.pickle', 'wb'))
pickle.dump(LRmodel, open('Sentiment-LR.pickle', 'wb'))
```

---

### 8️⃣ Loading Models

```python
def load_models():
    vectoriser = pickle.load(open(...))
    model = pickle.load(open(...))
    return vectoriser, model
```

---

### 9️⃣ Making Predictions

```python
def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    predictions = model.predict(textdata)
    return pd.DataFrame({'text': text, 'sentiment': predictions})
```

Example:
```python
text = ["I hate twitter", "May the Force be with you."]
df = predict(vectoriser, model, text)
```

---

## ✅ Conclusion  

You’ve now walked through a full sentiment analysis pipeline — from data loading and preprocessing to training, saving, and using machine learning models to classify sentiment in real time.

Whether for business insights or personal projects, this foundation empowers you to build smarter text-based applications.

--- 

🎥 **Watch the full code explanation on YouTube**: *Coming Soon!*  
📊 **Experiment with the code and tune models for even better accuracy!*

Happy Coding & Analyzing!

---