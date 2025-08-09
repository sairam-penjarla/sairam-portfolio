# 🦜🔗 Building AI Agents Using LangGraph: Part 7 — Handling Dynamic Inputs and Human Interrupts

Welcome to Part 7 of our LangGraph series! In Part 6, we explored enhanced state management with multiple schemas. Now, we will focus on how to handle **dynamic inputs** and integrate **human interrupts** in LangGraph workflows. This will make your AI agents more interactive and responsive, allowing them to adapt to real-time changes. Let’s see how we can implement this functionality!

---

## 🌟 What's New?

In this post, we’ll explore:
- Handling **dynamic user inputs** (e.g., real-time questions).
- Integrating **human interrupts** to modify the flow based on user inputs.
- Binding tools to LangChain’s **Groq model** for efficient AI responses.

---

## 🎯 The Goal

We’ll build a graph that:
- Accepts **human input** (e.g., “Who is sairam?”).
- Uses **predefined tools** to fetch a person’s details and location.
- Updates the state dynamically based on each user query.

The ultimate aim is to create a **real-time interactive AI system** that responds to user inputs and adapts accordingly.

---

## 💻 Code Walkthrough

Here’s how you can implement dynamic inputs and human interrupts:

### Define the State and Tools

```python
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Define the state for the graph
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Tool: Person Information
def get_person_details(person_name: str) -> str:
    """Retrieve details of a person."""
    return f"{person_name} is a DevOps Engineer."

# Tool: Location Information
def get_person_location(person_name: str) -> str:
    """Retrieve location of a person."""
    return f"{person_name} lives in Bangalore."

# Initialize ChatGroq model
llm = ChatGroq(model="llama-3.1-8b-instant")
# Bind tools to the Groq model
tools = [get_person_details, get_person_location]
llm_with_tools = llm.bind_tools(tools)
```

### Define the Assistant Node

```python
def assistant(state: MessagesState):
    sys_msg = SystemMessage("You are a helpful assistant who answers questions accurately using the available tools.")
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}
```

### Build the Graph

```python
def ask_human_input(prompt: str) -> str:
    """Helper function to prompt user input."""
    return input(f"{prompt}\n> ")

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools=tools))
# Add graph edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile the graph with human interrupts
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["assistant"],
    interrupt_after=[],
)
```

### Simulate Interaction

```python
# Simulate interaction
thread = {"configurable": {"thread_id": "1"}}
# Initialize state with a message
graph.invoke({"messages": [HumanMessage("Who is sairam?")]}, thread)
# Ask for user input and update the state
user_question = ask_human_input("Your question: Who is sairam?\nDo you want to ask anything else about sairam?")
user_message = HumanMessage(content=user_question)
# Update state with the new user message
graph.update_state(thread, {"messages": [user_message]})
# Stream the result
result = graph.stream(None, thread, stream_mode="values")
# Display the result
for event in result:
    event["messages"][-1].pretty_print()
```

---

## 🖥️ Output

**First Interaction**:

```
Your question: Who is sairam?
Do you want to ask anything else about sairam?
> No

================================ Human Message =================================
No
================================== AI Message ==================================
Tool Calls:
  get_person_details (call_ve92)
 Call ID: call_ve92
  Args:
    person_name: sairam
================================= Tool Message =================================
Name: get_person_details
sairam is a DevOps Engineer.
```

**Second Interaction**:

```
Your question: Who is sairam?
Do you want to ask anything else about sairam?
> Where is sairam living?

================================ Human Message =================================
Where is sairam living?
================================== AI Message ==================================
Tool Calls:
  get_person_location (call_epdv)
 Call ID: call_epdv
  Args:
    person_name: sairam
================================= Tool Message =================================
Name: get_person_location
sairam lives in Bangalore.
```

---

## 🛠️ How It Works

### Tools:
- **get_person_details**: Retrieves a person’s job role.
- **get_person_location**: Retrieves a person’s location.

### State Management:
- **MessagesState** tracks the conversation state, including user messages and AI responses.

### Human Interrupts:
- The graph dynamically waits for human input (e.g., “Who is sairam?”).
- After receiving the input, the system updates the state and responds in real time.
  
### Graph Flow:
- The graph starts at the assistant node, processes input, invokes tools, and updates the state as new inputs are received.

---

## 💬 Example Interaction

**System Message**:  
"You are a helpful assistant who answers questions accurately using the available tools."

**User Question**:  
*"Who is sairam?"*

**Assistant Response**:  
*"sairam is a DevOps Engineer."*

**Next Question**:  
The system prompts the user:  
*"Do you want to ask anything else about sairam?"*

---

## ✅ What’s Next?

**In Part 8**, we’ll optimize our LangGraph agent to gather information faster using **parallel node execution**. This will allow your agent to run tasks simultaneously and become much more efficient.