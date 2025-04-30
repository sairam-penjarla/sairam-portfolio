# Module 15: Natural Language Processing (NLP) - Named Entity Recognition (NER) using SpaCy, Prodigy, and Other Tools

## What is Named Entity Recognition (NER)?

**Named Entity Recognition (NER)** is a subtask of **Information Extraction** that seeks to locate and classify named entities in text into predefined categories such as:

- **Person names** (e.g., "John Doe")
- **Organizations** (e.g., "Google")
- **Locations** (e.g., "New York")
- **Dates and times** (e.g., "January 1st, 2025")
- **Monetary values** (e.g., "$1000")
- **Other entities** such as percentages, addresses, etc.

NER is useful in a wide variety of applications, such as **question answering**, **document summarization**, and **relationship extraction**. It helps machines understand the content and context of the text, enabling a deeper semantic understanding.

### Key Concepts in NER:

- **Entity Recognition**: Identifying the entity in a sentence.
- **Entity Classification**: Categorizing the recognized entity into predefined classes.
- **Contextual Understanding**: Recognizing that the same word may represent different entities based on context.

---

## Section 15.1: Named Entity Recognition with SpaCy

### SpaCy Overview

**SpaCy** is one of the most popular libraries for NLP tasks, including NER. It is fast, efficient, and comes with pre-trained models that can identify a wide range of entities out-of-the-box. It also provides functionality to train custom NER models if your dataset contains specialized entities that aren't covered by the pre-trained models.

### Code Example 1: Using SpaCy for NER

**Importing Libraries**

```python
import spacy

```

**Explanation**

- We import **SpaCy**, a robust library for NLP that provides pre-trained models for tasks such as tokenization, part-of-speech tagging, and named entity recognition.

---

**Loading a Pre-trained Model and Using NER**

```python
# Load the pre-trained SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Apple is looking at buying U.K. startup for $1 billion on January 1st, 2025."

# Process the text
doc = nlp(text)

# Print the named entities in the text
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

```

**Explanation**

- We load the **pre-trained English model** `en_core_web_sm` that SpaCy offers.
- The text is processed using **nlp()**, and it returns a **Doc** object containing the processed tokens and entities.
- The `.ents` attribute of the **Doc** object contains the named entities identified in the text.
- We loop through the entities and print each entity's text and label (e.g., **PERSON**, **ORG**, **GPE**, etc.).

---

### Output for the Example Text

**Output**

```
Apple (ORG)
U.K. (GPE)
$1 billion (MONEY)
January 1st, 2025 (DATE)

```

**Explanation**

- **Apple** is identified as an organization (**ORG**).
- **U.K.** is identified as a geopolitical entity (**GPE**).
- **$1 billion** is identified as a monetary value (**MONEY**).
- **January 1st, 2025** is identified as a date (**DATE**).

This output shows how SpaCy automatically identifies and classifies the entities in the text.

---

## Section 15.2: Named Entity Recognition with Prodigy

### Prodigy Overview

**Prodigy** is an annotation tool that enables users to annotate large amounts of data quickly. It's particularly useful for training custom NER models or for improving existing ones by adding examples of entities that the model might not have recognized correctly.

Prodigy is designed to be intuitive and fast, and it provides several pre-built workflows, including NER annotation, which can be used for labeling and training a model.

### Steps to Perform NER with Prodigy

1. **Install Prodigy**: Prodigy is a paid tool, but you can start by requesting a trial from the official website.
    
    ```bash
    pip install prodigy
    
    ```
    
2. **Use Prodigy to Annotate Text**: Prodigy allows users to annotate text interactively by presenting sentences to label. You can run the following command to start a NER annotation session.

```bash
prodigy ner.manual your_dataset en_core_web_sm ./data.jsonl

```

- **ner.manual**: This is the Prodigy recipe for manual NER annotation.
- **your_dataset**: This is the name of the dataset where the annotations will be stored.
- **en_core_web_sm**: This is the pre-trained SpaCy model used as a base for entity recognition.
- **./data.jsonl**: This file contains the text that will be presented for annotation.

After running the above command, Prodigy will display sentences and ask you to label entities. You can add new labels or confirm existing ones. After completing the annotation, you can save the annotations for later use in model training.

### Training a Custom NER Model in Prodigy

Once you've annotated the data, you can use it to **train a custom NER model** in SpaCy. Here’s an outline of the steps:

1. **Load the annotated data** into SpaCy.
2. **Train the model** on your custom annotations.

```python
import spacy
from spacy.training import Example

# Load the blank English model
nlp = spacy.blank("en")

# Load annotated data (assumed to be saved as a JSONL file)
annotations = []
with open("annotations.jsonl", "r") as file:
    for line in file:
        annotations.append(line)

# Convert annotations to SpaCy Examples and train the model
# (This step can be more complex depending on the format)
examples = [Example.from_dict(doc, annotation) for doc, annotation in annotations]

# Train the NER model
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner, last=True)

# Add labels to the NER component
for example in examples:
    for ent in example.annotations.get("entities"):
        ner.add_label(ent[2])  # Assuming `ent[2]` is the label

# Train the model (this example assumes you have a valid training loop)
nlp.begin_training()

```

---

## Section 15.3: Other NER Tools

### Stanford NER

The **Stanford NER** is another popular tool for performing named entity recognition. It is built using Java and comes with pre-trained models for English, Arabic, and other languages. You can use the Python wrapper, **StanfordNLP**, to integrate Stanford NER with Python.

### Code Example with Stanford NER

```python
from stanfordnlp.server import CoreNLPClient

# Start the CoreNLP client
with CoreNLPClient(annotators=["ner"], timeout=30000, memory="16G") as client:
    text = "Elon Musk is the CEO of Tesla, and he lives in California."
    ann = client.annotate(text)

    # Print named entities
    for sentence in ann.sentence:
        for token in sentence.token:
            print(f"{token.word}: {token.ner}")

```

**Explanation**

- **CoreNLPClient** starts a server to interact with the Stanford NER.
- We annotate the text and print the named entities and their labels.

---

### Output for Stanford NER Example

```
Elon: PERSON
Musk: PERSON
Tesla: ORGANIZATION
California: LOCATION

```

---