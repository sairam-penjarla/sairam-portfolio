# 🦜🔗 Building AI Agents Using LangGraph: Part 3 — Decision-Making with Routers

Welcome to Part 3 of our LangGraph series! In Part 2, we learned how to integrate tool-calling LLMs. Now, we will introduce **routers**, which allow your AI agent to **dynamically decide** how to respond. Whether it’s responding directly or invoking external tools, this part will guide you through building an intelligent decision-making agent.

---

## 🆕 What’s New in Part 3?

- **Tool Nodes**: Nodes that handle external tool invocation.
- **Conditional Edges**: Edges based on model output.
- **Loops**: Allow repeated back-and-forth between model and tools.

---

## 🧑‍💻 The Code

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

# Define the shared message state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Tool: Person Information Lookup
def fetch_person_details(name: str) -> str:
    """Returns job info for a given person name."""
    return f"{name} is a senior cloud architect specializing in distributed systems."

# Initialize the model and bind the tool
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools([fetch_person_details])

# Node function to invoke the LLM
def route_with_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

---

## 🧭 Building the Router Graph

```python
# Build the state graph
builder = StateGraph(MessagesState)

# Add the LLM node and tool node
builder.add_node("route_with_llm", route_with_llm)
builder.add_node("tools", ToolNode([fetch_person_details]))

# Define graph edges
builder.add_edge(START, "route_with_llm")
builder.add_conditional_edges("route_with_llm", tools_condition)
builder.add_edge("tools", END)

# Compile the final graph
graph = builder.compile()
```

This graph enables conditional decision-making. If the model decides a tool is needed, execution flows to the `ToolNode`.

---

## 🧪 Test the Agent

```python
# Test 1: Tool-triggering query
print("\n=== Test Case 1: Tool Invocation ===")
out1 = graph.invoke({"messages": [HumanMessage(content="Who is Alice?")]})
for msg in out1["messages"]:
    msg.pretty_print()

# Test 2: No tool required
print("\n=== Test Case 2: Direct LLM Response ===")
out2 = graph.invoke({"messages": [HumanMessage(content="Good morning!")]}
)
for msg in out2["messages"]:
    msg.pretty_print()
```

---

## 🧾 Sample Output

```
=== Test Case 1: Tool Invocation ===
Human: Who is Alice?
AI (Tool): Alice is a senior cloud architect specializing in distributed systems.

=== Test Case 2: Direct LLM Response ===
Human: Good morning!
AI: Good morning! How can I help you today?
```

---

## ✅ What’s Next?

**Next in Part 4**, we’ll evolve our router into a **fully dynamic agent** by applying the **ReAct** (Reason + Act) architecture. Your agent will learn to think, act, and reason in a loop, just like a real assistant.