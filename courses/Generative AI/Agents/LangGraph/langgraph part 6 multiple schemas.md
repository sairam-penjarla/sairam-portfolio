# 🦜🔗 Building AI Agents Using LangGraph: Part 6 — Using Multiple Schemas for Enhanced State Management

Welcome to Part 6 of our LangGraph series! In Part 5, we added memory to our agent. Now, we will explore how to use **multiple schemas** for enhanced state management. By structuring data flow more clearly, we can create modular and maintainable systems that improve the organization of our workflows. Let’s take a closer look at this powerful feature.

---

## 🔍 Why Multiple Schemas?

Using multiple schemas brings many benefits:
- **Modular Workflows**: Clearly define different stages of processing (input, intermediate, and output).
- **Data Validation**: Ensure correct formats at each stage.
- **Improved Debugging**: Isolate problems to specific stages for easier troubleshooting.

---

## 💡 Use Case: Generating Dynamic Responses

Let's build a graph where we input a person's name and get a **formatted message** with their job role and location.

For example, input: *"sairam"* should output:  
**"sairam is working as a DevOps Engineer at Bangalore."** 🏙️

### Step-by-Step Implementation:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Define the schemas
class InputState(TypedDict):
    name: str  # Input: Person's name

class OutputState(TypedDict):
    response: str  # Final output: Formatted response

class OverallState(TypedDict):
    name: str       # Intermediate: Person's name
    response: str   # Intermediate: Job description
    location: str   # Intermediate: Location information
```

### Graph Nodes

1. **Initial Node**: Process the input and generate initial responses.

```python
def initial_node(state: InputState):
    """Processes the input name and generates initial response."""
    return {
        "response": f"{state['name']} is working as a DevOps Engineer", 
        "location": "Bangalore"
    }
```

2. **Final Node**: Combine the response and location into the final message.

```python
def final_node(state: OverallState) -> OutputState:
    """Combines response and location into a final output."""
    return {"response": f"{state['response']} at {state['location']}"}
```

---

### Creating the Graph:

```python
graph = StateGraph(OverallState, input=InputState, output=OutputState)
graph.add_node("initial_node", initial_node)  # First node
graph.add_node("final_node", final_node)      # Final node

# Define graph flow
graph.add_edge(START, "initial_node")  # Start to initial processing
graph.add_edge("initial_node", "final_node")  # Intermediate to final processing
graph.add_edge("final_node", END)  # End the flow

# Compile and test the graph
graph = graph.compile()
result = graph.invoke({"name": "sairam"})
print(result["response"])  # Output the final response
```

---

## 🖥️ Output:

```
sairam is working as a DevOps Engineer at Bangalore
```

### The Flow Breakdown:

1. **Input Schema** (`InputState`):  
   - Captures the initial data (e.g., `{"name": "sairam"}`).

2. **Intermediate Schema** (`OverallState`):  
   - Adds context (e.g., `{"name": "sairam", "response": "...", "location": "Bangalore"}`).

3. **Output Schema** (`OutputState`):  
   - Filters and outputs the final, formatted response (e.g., `{"response": "sairam is working as a DevOps Engineer at Bangalore"}`).

---

## 🧐 Why Use Multiple Schemas?

- **Modularity**: Each stage is independent, making it easier to manage and debug.
- **Data Validation**: Automatically ensures the right format for data at each step.
- **Maintainability**: The system is more scalable and easier to extend as the project grows.

---

## ✅ What’s Next?

**In Part 7**, we’ll focus on handling **dynamic inputs** and adding support for **human interrupts**. You’ll learn how to make your agents more flexible and interactive in real-time conversations.