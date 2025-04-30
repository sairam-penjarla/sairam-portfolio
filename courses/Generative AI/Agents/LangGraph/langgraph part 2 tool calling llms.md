# 🦜🔗 Building AI Agents with LangGraph: Part 2 — Introducing Tool-Calling LLMs!

Welcome to Part 2 of our LangGraph series! In Part 1, we built a basic sentiment classification agent. Now, we will take a step forward by introducing tool-calling language models (LLMs) into our LangGraph workflows. This integration enables real-time information retrieval and dynamic behavior, allowing the agent to perform a wider range of tasks. Let’s explore this new feature!

---

## 🛠️ Why Use Tool-Calling?

Language models are powerful, but their abilities can be vastly enhanced by connecting them with external tools. This allows agents to:

- Retrieve live data (e.g., user profiles, weather, financial stats).
- Perform external actions (e.g., database lookups, API calls).
- Provide more accurate and relevant responses based on current context.

---

## 🌦️ Example: Person Info Lookup Agent

We’ll build an agent that can answer questions like:  
**“Who is Alice?”**  
The agent will use a tool that simulates retrieving a person’s information.

---

## 📋 Define the State

```python
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

This `MessagesState` holds a list of messages exchanged during the conversation.

---

## 🔧 Define the Tool

```python
def fetch_person_info(name: str) -> str:
    return f"{name} is a senior software engineer with 5 years of experience."
```

This function mimics an external API or database call.

---

## 💬 Set Up the Tool-Calling LLM

```python
from langchain_groq import ChatGroq

# Load the model
llm = ChatGroq(model="llama3-70b-8192")

# Bind the tool to the model
llm_with_tools = llm.bind_tools([fetch_person_info])
```

The model is now capable of recognizing and calling the `fetch_person_info` tool.

---

## 🧠 Create the Tool-Calling Node

```python
def invoke_tool_via_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

This node invokes the LLM, which can optionally call the tool based on user input.

---

## 🌐 Build the LangGraph Workflow

```python
from langgraph.graph import StateGraph, START, END

# Initialize the graph
builder = StateGraph(MessagesState)

# Add the tool-calling node
builder.add_node("invoke_tool", invoke_tool_via_llm)

# Define the edges
builder.add_edge(START, "invoke_tool")
builder.add_edge("invoke_tool", END)

# Compile the graph
graph = builder.compile()
```

This creates a simple graph that invokes the LLM once, passing the messages and handling any tool calls.

---

## 🧪 Run the Graph

```python
from langchain_core.messages import HumanMessage

# Test 1: With a tool-callable query
print("\n=== Test Case 1: Query about a person ===")
output1 = graph.invoke({"messages": [HumanMessage(content="Who is Alice?", name="User")]})
for msg in output1["messages"]:
    msg.pretty_print()

# Test 2: General greeting
print("\n=== Test Case 2: Greeting ===")
output2 = graph.invoke({"messages": [HumanMessage(content="Hi there!")]})
for msg in output2["messages"]:
    msg.pretty_print()
```

---

## 🧾 Sample Output

```
=== Test Case 1: Query about a person ===
============================== Human Message ==============================
Name: User

Who is Alice?
=============================== AI Message ===============================
Tool Calls:
  fetch_person_info
Call ID: call_tool_1
Args:
  name: Alice

Alice is a senior software engineer with 5 years of experience.

=== Test Case 2: Greeting ===
============================== Human Message ==============================

Hi there!
=============================== AI Message ===============================

Hello! How can I assist you today?
```

---

## ✅ What’s Next?

**In Part 3**, we’ll introduce a new concept: **routers**. These allow your AI agent to dynamically decide how to handle different inputs—either responding directly or invoking external tools. Let’s give our agent some decision-making power!