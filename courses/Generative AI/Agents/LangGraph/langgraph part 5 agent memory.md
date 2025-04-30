# 🦜🔗 Building AI Agents Using LangGraph: Part 5 — Adding Memory to the Agent

Welcome back to Part 5 of our LangGraph series! In Part 4, we built a dynamic agent that could act, observe, and reason. Now, we will enhance the agent to **remember** previous interactions and retain context across multiple turns using **LangGraph’s MemorySaver**. This will enable the agent to have a more coherent and persistent memory of past events.

---

## 🔁 Recap: Our Stateless Agent

Without memory, each interaction was isolated:

### ❌ Problem:
> Ask: “Tell me about sairam?”  
> Follow-up: “Is sairam a DevOps Engineer?”  
>  
> The agent forgets the first query and treats the second independently — resulting in confusion or redundant tool calls.

---

## ⚙️ Step 1: Basic Stateless Agent Setup

Here’s the original structure:

```python
from typing_extensions import TypedDict, Annotated
from typing import Any
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def get_person_details(person_name: str) -> str:
    return f"{person_name} is a DevOps Engineer."

def get_person_location(person_name: str) -> str:
    return f"{person_name} lives in Bangalore."

llm = ChatGroq(model="llama-3.3-70b-versatile")
tools = [get_person_details, get_person_location]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant...")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()
```

---

## 🧪 Example Without Memory

```python
# Test 1
messages = [HumanMessage(content="Tell me about sairam?")]
result = react_graph.invoke({"messages": messages})

# Test 2
messages = [HumanMessage(content="Is sairam a Devops Enginer?")]
result = react_graph.invoke({"messages": messages})
```

### 🚫 Outcome:
> Agent repeatedly asks tools for the same info or fails to connect the dots.

---

## ✅ Step 2: Add Memory with `MemorySaver`

LangGraph includes a built-in checkpointing system. Let's plug it in.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

### 🧵 Use `thread_id` for Persistent Sessions

```python
config = {"configurable": {"thread_id": "1"}}
```

### 🗣️ Interact with Context

```python
# First message
messages = [HumanMessage(content="Tell me about sairam?")]
result = react_graph_memory.invoke({"messages": messages}, config)

# Second message (same thread)
messages = [HumanMessage(content="Is sairam a Devops Enginer?")]
result = react_graph_memory.invoke({"messages": messages}, config)

for message in result['messages']:
    message.pretty_print()
```

---

### ✅ Outcome with Memory:

```
================================ Human Message ================================
Is sairam a Devops Enginer?

============================= Tool Call: get_person_details ====================
sairam is a DevOps Engineer.

============================= Tool Call: get_person_location ===================
sairam lives in Bangalore.

============================= Final AI Message ================================
Based on the details, sairam is a DevOps Engineer and lives in Bangalore.
```

The agent **remembers** the subject "sairam" and uses **historical context** to answer seamlessly. 🧠✅

---

## ✨ Benefits of Adding Memory

- **Context Retention**: Understands ongoing dialogue threads.
- **Reduced Redundancy**: Prevents unnecessary reprocessing of previously answered queries.
- **More Natural Conversations**: Mirrors how real assistants behave.

---

## ✅ What’s Next?

**Coming up in Part 6**, we’ll learn how to manage complex agent states using **multiple schemas**. This allows for better modularity and structure in your workflows—perfect for scaling larger applications.