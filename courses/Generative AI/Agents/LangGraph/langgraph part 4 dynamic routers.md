# 🦜🔗 Building AI Agents Using LangGraph: Part 4 — Dynamic Routers with Tool Integration

Welcome to Part 4 of our LangGraph series! In Part 3, we explored decision-making using routers. Now, we’ll enhance our router into a fully dynamic agent using the **ReAct** (Reason + Act) architecture. This approach allows the agent to continuously think, act, and reason using multiple tools. Let’s dive into building a more intelligent agent!

---

## 🌟 What’s New?

Our agent now supports:

- **Acting 🤹**: Chooses whether to invoke one or more tools.
- **Observing 🕵️**: Passes tool outputs back to the model.
- **Reasoning 🤔**: Decides what to do next — call more tools or respond directly.

This structure enables continuous loops until the assistant generates a final, fully informed answer.

---

## 🗺️ Adding Location Information

Let’s upgrade our assistant to not only know **what someone does**, but also **where they live**.

### 📦 Setup

```python
from typing_extensions import TypedDict, Annotated
from typing import Any
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

# Shared message state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

---

### 🛠️ Define Tools

```python
# Tool 1: Person Information
def get_person_details(person_name: str) -> str:
    """Retrieve details of a person."""
    return f"{person_name} is a DevOps Engineer."

# Tool 2: Location Information
def get_person_location(person_name: str) -> str:
    """Retrieve location of a person."""
    return f"{person_name} lives in Bangalore."
```

---

### 🤖 Model Setup

```python
# Use a more capable Groq model for reasoning
llm = ChatGroq(model="llama-3.3-70b-versatile")
tools = [get_person_details, get_person_location]
llm_with_tools = llm.bind_tools(tools)

# Assistant node that reasons and decides next steps
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

---

### 🔁 Build the ReAct Graph

```python
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Flow control
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")  # Feedback loop for iterative reasoning

# Compile the agent graph
react_graph = builder.compile()
```

---

## 🚀 Test Run

```python
messages = [HumanMessage(content="Tell me about sairam. Where does she live?")]
result = react_graph.invoke({"messages": messages})

# Output
for msg in result["messages"]:
    msg.pretty_print()
```

---

### 🖥️ Expected Output

```
================================ Human Message ================================
Tell me about sairam. Where does she live?

=============================== AI Message (Tool Call) ========================
Calling: get_person_details

============================== Tool Message Output ============================
sairam is a DevOps Engineer.

=============================== AI Message (Tool Call) ========================
Calling: get_person_location

============================== Tool Message Output ============================
sairam lives in Bangalore.

================================ Final AI Message =============================
sairam is a DevOps Engineer and lives in Bangalore.
```

---

## ✅ What’s Next?

**In Part 5**, we’ll equip our AI agent with **memory**. By remembering previous interactions, your agent can hold context across turns and respond more intelligently. We’ll use LangGraph’s built-in memory features to make this happen.
