# 🦜🔗 Building AI Agents Using LangGraph: Part 8 — Implementing Parallel Node Execution for Efficient Context Gathering

Welcome to Part 8 of our LangGraph series! In Part 7, we explored dynamic inputs and human interrupts. Now, we’ll dive into **parallel node execution**, allowing your agent to gather context from different sources simultaneously. This method enhances efficiency and enables the agent to combine information more quickly and effectively. Let’s learn how to implement this technique!

---

### What’s New? 🌟

In this post, we’ll cover:
- **Reducers**: How they work and why they’re essential.
- **State Updates**: Using reducers to append, concatenate, or modify state data effectively.
- **Practical Application**: Managing message-related states using LangGraph’s built-in reducers like `add_messages`.

By the end, you’ll have a flexible, state-aware AI graph that tracks and updates its state while retaining context. 🤖

---

### The Goal 🎯

We aim to create a system where:
1. Values returned from each node append to the state rather than overwrite it.
2. Message states are dynamically updated using prebuilt reducers.
3. We leverage the `MessagesState` class for smoother integration of message-related workflows.

---

### Code Walkthrough 💻

Here’s how to efficiently implement reducers for state management:

---

#### 1️⃣ **Define State and Reducers**

We’ll define a state schema and reducers that handle the appending of lists and messages. For this example, we use Python's `operator.add` to concatenate lists, and LangGraph’s built-in `add_messages` to append messages.

```python
from operator import add
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Define state schema
class State(TypedDict):
    count: Annotated[list[int], add]  # Concatenates integers
    log: Annotated[list[AnyMessage], add_messages]  # Concatenates messages
```

---

#### 2️⃣ **Create Node Functions**

Next, we’ll create node functions that update both the `count` and `log` keys in the state. Each node will append new values to these lists.

```python
def node_1(state):
    return {
        "count": [state['count'][-1] + 1],
        "log": [state['log'][-1]] + [HumanMessage("I'm from Node 1")]
    }

def node_2(state):
    return {
        "count": [state['count'][-1] + 1],
        "log": [state['log'][-1]] + [HumanMessage("I'm from Node 2")]
    }

def node_3(state):
    return {
        "count": [state['count'][-1] + 1],
        "log": [state['log'][-1]] + [HumanMessage("I'm from Node 3")]
    }
```

---

#### 3️⃣ **Build and Connect the Graph**

Now, we’ll create the state graph and connect the nodes. The edges define the flow of the graph, ensuring that state updates pass through the nodes in the right order.

```python
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Define edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Compile the graph
graph = builder.compile()
```

---

#### 4️⃣ **Invoke the Graph**

Finally, let’s initialize the state and invoke the graph. This will allow us to see how the state evolves as it passes through the nodes.

```python
# Initial state
initial_state = {"count": [1], "log": [HumanMessage("Start")]}

# Invoke the graph
result = graph.invoke(initial_state)

# Display results
print("================================ Add Operator =================================")
print(result['count'])  # Tracks state updates for `count`

print("\n================================ Add Message =================================")
print(result['log'])  # Tracks message state
```

---

### Output 📝

Here’s what we get when we invoke the graph:

#### Add Operator (count)

```
================================ Add Operator =================================
[1, 2, 3, 3]
```

The `count` key tracks how the values accumulate through each node. The sequence begins with `[1]`, and each node adds `1` to the last value in the list:
- **Node 1**: Adds 1 → `[1, 2]`
- **Node 2**: Adds 1 → `[1, 2, 3]`
- **Node 3**: Adds 1 → `[1, 2, 3, 3]`

#### Add Message (log)

```
================================ Add Message =================================
[
    HumanMessage(content='Start', ...),
    HumanMessage(content="I'm from Node 1", ...),
    HumanMessage(content="I'm from Node 2", ...),
    HumanMessage(content="I'm from Node 3", ...)
]
```

The `log` key accumulates messages as the state flows through the graph:
- **Node 1**: Adds `"I'm from Node 1"`
- **Node 2**: Adds `"I'm from Node 2"`
- **Node 3**: Adds `"I'm from Node 3"`

---

### Key Concepts 🔑

#### **Reducers**
Reducers define how state updates are performed. In this example:
- **`operator.add`**: Used to concatenate lists, like the `count` key.
- **`add_messages`**: A specialized reducer for appending messages, such as those in the `log` key.

#### **MessagesState Shortcut**
The `MessagesState` class simplifies handling messages by:
- Automatically applying the `add_messages` reducer.
- Managing message-related workflows efficiently.

---

## ✅ What’s Next?

**Coming up in Part 9**, we’ll integrate **real-time data from the web and external APIs** to give our agent a richer understanding of the world. This enhances the relevance and intelligence of its responses.