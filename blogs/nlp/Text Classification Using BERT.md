# **Text Classification Using BERT: A Practical Guide for Intermediate NLP Practitioners**  
📅 *June 27, 2024*

In this post, we'll explore how to build a text classification pipeline using BERT, with a special focus on classifying **German tweets** for different types of abusive language. We'll break down both the theory and the code, guiding you through every step from data preparation to model deployment.

---

## 🔗 GitHub Repository

To get the complete source code, required files (like `requirements.txt` and dataset CSVs), and to try it yourself, head over to the GitHub repo:

👉 [https://github.com/sairam-penjarla/NLP-Text-Classification-BERT](https://github.com/sairam-penjarla/NLP-Text-Classification-BERT)

---

## 🧠 Understanding the Theory

### What is BERT?

BERT (Bidirectional Encoder Representations from Transformers) is a language representation model developed by Google. Unlike traditional NLP models, BERT understands the context of a word based on **both** the left and right surroundings, enabling it to grasp subtle language nuances.

### Why Use BERT for Text Classification?

BERT is ideal for tasks like sentiment analysis, spam detection, and abuse detection because it can be **fine-tuned** on labeled datasets, adapting to domain-specific language patterns.

---

## 🔍 Project Overview

We'll classify German tweets into one of several **abusive language categories**, using a pre-trained German BERT model. Our approach involves:

1. Data preprocessing
2. Model training
3. Evaluation and saving
4. Reloading for inference

---

## 🧰 Step-by-Step Code Walkthrough

### 1. Import Required Libraries

```python
import os
import torch
import tarfile
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
```

- `simpletransformers` simplifies working with Hugging Face Transformers.
- We also use `pandas`, `sklearn`, and `PyTorch` for handling data and computations.

---

### 2. Load and Preprocess the Data

```python
class_list = ['INSULT','ABUSE','PROFANITY','OTHER', 'EXPLICIT', 'IMPLICIT']

df1 = pd.read_csv('Shared-Task-2019_Data_germeval2019.training_subtask1_2.txt', sep='\\t', lineterminator='\\n', encoding='utf8', names=["tweet", "task1", "task2"])
df2 = pd.read_csv('Shared-Task-2019_Data_germeval2019.training_subtask3.txt', sep='\\t', lineterminator='\\n', encoding='utf8', names=["tweet", "task1", "task2"])

df = pd.concat([df1, df2])
df['task2'] = df['task2'].str.replace('\\r', "")
df['pred_class'] = df['task2'].apply(lambda x: class_list.index(x))
df = df[['tweet', 'pred_class']]
```

We concatenate two labeled datasets and map each textual class label to a numeric index.

---

### 3. Split the Dataset

```python
train_df, test_df = train_test_split(df, test_size=0.10)
```

This ensures we evaluate the model on unseen data (10% test split).

---

### 4. Set Up and Train the Model

```python
train_args = {
    "reprocess_input_data": True,
    "fp16": False,
    "num_train_epochs": 4
}

model = ClassificationModel(
    "bert", "distilbert-base-german-cased",
    num_labels=len(class_list),
    args=train_args,
    use_cuda=torch.cuda.is_available()
)

model.train_model(train_df)
```

We're using the `"distilbert-base-german-cased"` model—a distilled and faster version of BERT optimized for German.

---

### 5. Evaluate the Model

```python
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')

result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
```

We assess the model's performance using both **accuracy** and **micro-averaged F1 score**.

---

### 6. Save the Model

```python
def pack_model(model_path='', file_name=''):
    files = [files for root, dirs, files in os.walk(model_path)][0]
    with tarfile.open(file_name + '.tar.gz', 'w:gz') as f:
        for file in files:
            f.add(f'{model_path}/{file}')
            
pack_model('output_path', 'model_name')
```

Save the trained model for future use—no need to retrain from scratch!

---

### 7. Load the Model

```python
def unpack_model(model_name=''):
    tar = tarfile.open(f"{model_name}.tar.gz", "r:gz")
    tar.extractall()
    tar.close()

unpack_model('model_name')
```

You can now reuse the trained model by extracting it from storage.

---

### 8. Make Predictions

```python
model = ClassificationModel(
    "bert", 'path_to_model/',
    num_labels=4,
    args=train_args
)

test_tweet1 = "Meine Mutter hat mir erzählt, dass mein Vater einen Wahlkreiskandidaten nicht gewählt hat, weil der gegen die Homo-Ehe ist"
test_tweet2 = "Frau #Böttinger meine Meinung dazu ist sie sollten uns mit ihrem Pferdegebiss nicht weiter belästigen #WDR"

predictions, _ = model.predict([test_tweet1])
print(class_list[predictions[0]])  # OUTPUT: OTHER

predictions, _ = model.predict([test_tweet2])
print(class_list[predictions[0]])  # OUTPUT: INSULT
```

Our model is now ready to classify any German tweet based on its content.

---

## ✅ Conclusion

In this guide, we walked through the full process of building a BERT-based text classification pipeline. From preprocessing German tweets to training, evaluating, and using the model for predictions—we've covered everything.

With the help of `simpletransformers`, working with transformer models becomes approachable even for intermediate practitioners.

---

**Try modifying this workflow** for other languages, classification tasks, or even multi-label classification. The possibilities are endless!