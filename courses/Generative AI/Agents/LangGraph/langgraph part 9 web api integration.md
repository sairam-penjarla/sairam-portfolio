# 🦜🔗 Building AI Agents Using LangGraph: Part 9 — Integrating Web and API Data for Enhanced Context

Welcome to Part 9 of our LangGraph series! In Part 8, we explored parallel node execution for efficient context gathering. In this part, we’ll integrate **web and API data** into our workflows to further enrich the agent’s context. By fetching real-time data, we can provide more accurate and relevant responses, improving the agent’s overall performance.

---

### What’s New? 🌟
In this post, we’ll cover:
- **Parallel Execution:** How to run nodes concurrently to gather information from multiple sources.
- **Context Management:** How to preserve context from multiple nodes without losing data.
- **Summarization with LLM:** Using LangGraph to synthesize information from various sources using an LLM.

### The Goal 🎯
The goal is to create a dynamic and responsive AI agent capable of processing data from various sources in parallel and summarizing it effectively.

### Code Walkthrough 💻
Let’s break down the implementation:

#### 1. Define the State and Context
We begin by defining a state schema that will allow us to store the query results from different sources like DuckDuckGo and Wikipedia.

```python
from operator import add
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchRun

# Define the state schema
class State(TypedDict):
    query: str  # User's query
    answer: str  # Generated answer
    retrieved_context: Annotated[list, add]  # Combined context
```

#### 2. Create Node Functions
Each node is responsible for fetching data from different sources or generating an answer based on the context gathered:

- **DuckDuckGo Search Node:**

```python
def search_duckduckgo(state):
    """Fetch information from DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    response = search.invoke(state['query'])
    return {"retrieved_context": [response]}
```

- **Wikipedia Search Node:**

```python
def search_wikipedia(state):
    """Fetch information from Wikipedia."""
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    response = wikipedia.run(state['query'])
    return {"retrieved_context": [response]}
```

- **Answer Generation Node:**

```python
def generate_answer(state):
    """Generate an answer using the retrieved context."""
    retrieved_context = state['retrieved_context']
    query = state['query']

    # Prompt Template
    prompt_template = f"""Based on the following context, provide an accurate and concise answer to the query:
    Query: {query}
    Context: {retrieved_context}"""

    # Generate the answer
    response = llm.invoke([SystemMessage(content=prompt_template)] +
                          [HumanMessage(content="Provide the response.")])
    return {"answer": response}
```

#### 3. Build and Connect the Graph
Next, we build a graph where the nodes for DuckDuckGo and Wikipedia feed into the answer-generation node:

```python
builder = StateGraph(State)

# Add nodes
builder.add_node("search_duckduckgo", search_duckduckgo)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Define edges
builder.add_edge(START, "search_duckduckgo")
builder.add_edge(START, "search_wikipedia")
builder.add_edge("search_duckduckgo", "generate_answer")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("generate_answer", END)

# Compile the graph
graph = builder.compile()
```

#### 4. Invoke the Graph
Finally, we invoke the graph with a query and observe how the state evolves:

```python
# Query
query = "How can AI agents improve observability in DevOps?"

# Invoke the graph
result = graph.invoke({"query": query})

# Display the generated answer
print(result['answer'].content)
```

### Output 📝
For the query "How can AI agents improve observability in DevOps?", the AI agent will:
- Retrieve real-time insights from DuckDuckGo.
- Fetch structured data from Wikipedia.
- Combine these contexts to generate a comprehensive and accurate response.

**Sample Answer:**

AI agents can improve observability in DevOps by:
1. **Enabling proactive management:** Gaining insights into the agents' workings, detecting anomalies, and preventing failures.
2. **Providing scalability and consistency:** Managing larger systems without proportional increases in human resources.
3. **Offering low-code tools:** Mapping user journeys and improving system uptime and user experience.
4. **Surfacing opinionated insights:** Democratizing observability and helping organizations better understand their systems.
5. **Enhancing control and flexibility:** Giving developers more control over AI agents, enhancing flexibility and usage.

By implementing these capabilities, AI agents can transform DevOps practices and improve overall system observability.

### Key Concepts 🔑

#### Parallel Node Execution
In this example, we used LangGraph to run two nodes in parallel: `search_duckduckgo` and `search_wikipedia`. Both nodes independently fetch relevant information from their respective sources based on the user’s query. This parallel approach speeds up data retrieval and provides a broader context for the agent, improving the final response.

#### Context Management with State
Rather than overwriting the state, we update the `retrieved_context` dynamically. The results from each node are appended to the state without losing any previous data, ensuring that the AI agent maintains all relevant context before synthesizing the final answer.

#### Summarization Using LLM
Once the context is gathered from both sources, we use a language model (ChatGroq) to combine and summarize the information. This allows the agent to process and synthesize the context from multiple sources, delivering a more accurate and comprehensive answer to the query.

#### Reducers and State Updates
LangGraph uses reducers to manage state updates. In this case, we used the `add` reducer to append the responses from DuckDuckGo and Wikipedia to the `retrieved_context`, ensuring both results are preserved without overwriting, allowing for a richer and more informed response.

---

## ✅ What’s Next?

**In Part 10**, we’ll introduce the idea of **subgraphs** to build **multi-agent systems**. Subgraphs are reusable and modular, making it easy to scale complex AI agents and delegate responsibilities between them.