# A Sentiment Classifier

Welcome to Part 1 of our LangGraph series! In this part, we will build a simple AI agent that classifies user input as either positive or negative sentiment using LangGraph. LangGraph is a powerful framework for constructing modular, stateful AI workflows with a graph-based structure. Let's dive in and start building!

---

## 📘 What Is LangGraph?

LangGraph is a Python library designed to manage complex decision-making processes by modeling them as directed graphs. Each **node** represents a processing step, and **edges** define transitions based on logic or conditions.

Think of it as an easy way to construct intelligent workflows—whether you're routing data, triggering responses, or managing state.

---

## 📋 Step 1: Define the State

LangGraph workflows pass around a state object between nodes. We’ll define a `State` class using Python’s `TypedDict` to clearly describe the structure:

```python
from typing_extensions import TypedDict

class State(TypedDict):
    message: str
    sentiment: str
```

- `message`: The user’s input text.
- `sentiment`: The classification result ("positive", "negative", or "neutral").

---

## 🧩 Step 2: Create Processing Nodes

Each node is a Python function that receives the state and returns an updated dictionary. Let’s define our sentiment analysis logic:

```python
def analyze_sentiment(state):
    print("--- Analyzing Sentiment ---")
    text = state["message"].lower()

    if any(word in text for word in ["great", "good", "happy"]):
        return {"sentiment": "positive"}
    elif any(word in text for word in ["bad", "terrible", "sad"]):
        return {"sentiment": "negative"}
    else:
        return {"sentiment": "neutral"}
```

We’ll also define responses based on the sentiment:

```python
def positive_response(state):
    print("--- Positive Response ---")
    return {"message": "Thanks for the positive feedback! 😊"}

def negative_response(state):
    print("--- Negative Response ---")
    return {"message": "Sorry to hear that. How can we do better? 😞"}
```

---

## 🔀 Step 3: Define Conditional Routing

We need a router function to decide which node to run after the sentiment analysis step.

```python
from typing import Literal

def sentiment_router(state) -> Literal["positive_response", "negative_response"]:
    return "positive_response" if state["sentiment"] == "positive" else "negative_response"
```

---

## 🌐 Step 4: Build the Graph

Let’s connect everything using LangGraph’s `StateGraph` class.

```python
from langgraph.graph import StateGraph, START, END

# Initialize the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("positive_response", positive_response)
builder.add_node("negative_response", negative_response)

# Define flow
builder.add_edge(START, "analyze_sentiment")
builder.add_conditional_edges("analyze_sentiment", sentiment_router)
builder.add_edge("positive_response", END)
builder.add_edge("negative_response", END)

# Compile the graph
graph = builder.compile()
```

---

## 🚀 Step 5: Run the Agent

Let’s test it with a sample input:

```python
result = graph.invoke({"message": "I had a great experience!"})
print(result)
```

**Output:**
```
--- Analyzing Sentiment ---
--- Positive Response ---
{'message': 'Thanks for the positive feedback! 😊'}
```

---

## 🧪 Test with Different Inputs

Try other messages like:

```python
graph.invoke({"message": "This was terrible service."})
graph.invoke({"message": "It was okay, not great or bad."})
```

---

## ✅ What’s Next?

**Coming up in Part 2**, we’ll explore how to integrate **tool-calling language models (LLMs)** into our LangGraph workflow. This will allow your agent to perform real-time lookups and make more dynamic decisions. Stay tuned!