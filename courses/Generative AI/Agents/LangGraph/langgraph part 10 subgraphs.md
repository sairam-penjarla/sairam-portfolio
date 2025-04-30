# 🦜🔗 Building AI Agents Using LangGraph: Part 10 — Leveraging Subgraphs for Multi-Agent Systems

Welcome to Part 10 of our LangGraph series! In Part 9, we explored integrating web and API data. Now, we will focus on **subgraphs**, a powerful feature that enables modularity and reuse of logic across different workflows. Subgraphs are especially useful in **multi-agent systems**, where different components need to interact while retaining their unique states. Let’s learn how to implement this concept in LangGraph!
s
---

The key challenge when working with subgraphs is how to manage communication between the **parent graph** and the **subgraph**. In this article, we’ll walk through two main scenarios:
1. When both the parent and subgraph share schema keys, making integration straightforward.
2. When the parent and subgraph have different schemas, requiring state transformations before or after invoking the subgraph.

Let’s take a deeper look at how LangGraph simplifies handling these cases.

### What’s New? 🌟
In this post, you’ll learn about:
- **Subgraphs:** How to create and use subgraphs for modular systems.
- **Communication Between Parent and Subgraphs:** How to pass state between them, especially when their schemas differ.
- **Multi-Agent Systems:** Leveraging subgraphs to build AI agents that perform multiple tasks in parallel while maintaining isolated states.

### The Goal 🎯
Our goal is to create an AI agent framework that scales by breaking down complex workflows into smaller, reusable components (subgraphs) while maintaining state integrity across various agents.

### Code Walkthrough 💻
Let’s walk through the process of defining, building, and using subgraphs in a multi-agent system:

#### 1. Define States for Each Subgraph
We begin by defining the state for each subgraph. Here, we create two subgraphs: one for tracking job details and another for tracking location details.

- **Job Subgraph:**
```python
class JobState(TypedDict):
    name: Annotated[list[str], add]
    job: str

def job_node(state):
    return {"job": "DevOps Engineer"}

def job_dummy_node(state):
    pass
```

- **Location Subgraph:**
```python
class LocationState(TypedDict):
    name: Annotated[list[str], add]
    location: str

def location_node(state):
    return {"location": "Bangalore"}

def location_dummy_node(state):
    pass
```

#### 2. Build the Subgraphs
Now, we define each subgraph by adding nodes and edges, and then compiling them into graphs.

- **Job Subgraph Construction:**
```python
job_builder = StateGraph(JobState)
job_builder.add_node("job_node", job_node)
job_builder.add_node("job_dummy_node", job_dummy_node)
job_builder.add_edge(START, "job_node")
job_builder.add_edge("job_node", "job_dummy_node")
job_builder.add_edge("job_dummy_node", END)
job_graph = job_builder.compile()
```

- **Location Subgraph Construction:**
```python
location_builder = StateGraph(LocationState)
location_builder.add_node("location_node", location_node)
location_builder.add_node("location_dummy_node", location_dummy_node)
location_builder.add_edge(START, "location_node")
location_builder.add_edge("location_node", "location_dummy_node")
location_builder.add_edge("location_dummy_node", END)
location_graph = location_builder.compile()
```

#### 3. Create the Parent Graph
The parent graph integrates the results from the Job and Location subgraphs and generates a final summary. We define nodes for the subgraphs and a final node to merge the results.

- **Parent Graph State and Nodes:**
```python
class ParentState(TypedDict):
    name: Annotated[list[str], add]
    job: str
    location: str
    details: str

def first_node(state: ParentState):
    return {"name": [state['name'][-1]]}

def last_node(state: ParentState):
    return {"details": f"{state['name'][-1]} is a {state['job']} at {state['location']}"}
```

#### 4. Add Subgraphs to Parent Graph
We now incorporate the Job and Location subgraphs into the parent graph, ensuring they contribute to the final result.

```python
entry_builder = StateGraph(ParentState)
entry_builder.add_node("first_node", first_node)
entry_builder.add_node("location_subgraph", location_graph)
entry_builder.add_node("job_subgraph", job_graph)
entry_builder.add_node("last_node", last_node)

entry_builder.add_edge(START, "first_node")
entry_builder.add_edge("first_node", "location_subgraph")
entry_builder.add_edge("first_node", "job_subgraph")
entry_builder.add_edge("location_subgraph", "last_node")
entry_builder.add_edge("job_subgraph", "last_node")
entry_builder.add_edge("last_node", END)
```

#### 5. Compile and Invoke the Graph
Finally, we compile and invoke the graph, passing the initial state and retrieving the result.

```python
graph = entry_builder.compile()
result = graph.invoke({"name": ["sairam"]})

print(result['details'])
```

### Output 📝
When executed, the system will output the following:
```
sairam is a DevOps Engineer at Bangalore
```
This output demonstrates how the parent graph integrates the results from both subgraphs (Job and Location), combining them into a comprehensive final result.

### Key Concepts 🔑

#### Subgraphs
Subgraphs allow us to break down complex workflows into smaller, manageable components. These components can be compiled and reused independently. In this example, the Job and Location subgraphs handle independent tasks (job and location details) and pass their results back to the parent graph for integration.

#### Communication Between Parent and Subgraphs
The parent graph communicates with its subgraphs by passing state between them. When subgraphs share the same schema, they can be directly added to the parent graph. If the schemas differ, we manage state transformations to align the data. In this case, each subgraph operates independently, and the results are merged in the parent graph.

#### Multi-Agent Systems
By building separate subgraphs for job and location, we create **multi-agent systems**, where each subgraph (agent) performs a specific task. The parent graph integrates the outputs of these agents to provide a comprehensive response. This modular approach enables scalable, flexible, and more efficient AI systems.

---

## ✅ What’s Next?

**Next in Part 11**, we’ll put all the pieces together in a **real-world project**—a Restaurant & Weather Recommendation System. You’ll see how everything from memory to decision-making and tool usage comes together.