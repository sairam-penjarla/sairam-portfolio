# 🦜🔗 Building AI Agents Using LangGraph: Part 11 — Building a Restaurant & Weather Recommendation System

Welcome to Part 11 of our LangGraph series! In this final part, we’ll take everything we’ve learned so far and apply it to a practical, real-world use case: a **Restaurant & Weather Recommendation System**. Imagine you’re planning a date night and asking an AI to suggest the best restaurants based on the event and weather. That’s exactly what we’ll build today! Let’s get started.

---

### What’s New? 🌟
In this post, you’ll learn about:
- **Multi-Agent Coordination:** How multiple AI agents (like weather and restaurant finders) can work together.
- **Tool Integration:** Using external APIs (DuckDuckGo for restaurants and OpenWeatherMap for weather) to fetch live data.
- **State Management:** Sharing and transforming data between subgraphs and parent graphs.
- **Real-World Application:** Creating a practical system for event-based recommendations.

### The Goal 🎯
We’re building a system where:
- A user asks for restaurant recommendations for a specific location and event.
- The system checks the weather forecast.
- It fetches top-rated restaurants for that event and location.
- Finally, the AI suggests whether to go out or stay in based on weather conditions.

Let’s dive into the code! 💻

---

### 1️⃣ Setting Up the Environment 🛠️

We’ll start by loading the necessary libraries and initializing the Groq model:

```python
import datetime
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Initialize the Groq chat model
llm = ChatGroq(model="llama-3.3-70b-versatile")
```

### 2️⃣ Defining States and Tools 🧠

We define states for both the weather and restaurant subgraphs.

```python
class WeatherState(TypedDict):
    messages: Annotated[list, add_messages]
    weather_report: str

class Restaurant(TypedDict):
    name: Annotated[str, ..., "Name of the restaurant"]
    address: Annotated[str, ..., "Address of the restaurant"]
    details: Annotated[str, ..., "Details of the restaurant"]

class RestaurantState(TypedDict):
    messages: Annotated[list, add_messages]
    location: str
    event: str
    date: str
    restaurants: list[Restaurant]
```

### 3️⃣ Building the Weather Subgraph 🌦️

The weather subgraph fetches real-time weather data and formats it.

```python
# Weather assistant node
def weather_assistant(state: WeatherState):
    return {"messages": [llm.invoke(state["messages"])]}

def weather_formatter(state: WeatherState):
    weather_report = ""
    for message in state['messages']:
        weather_report += message.content
    return {"weather_report": weather_report}

# Building the weather graph
weather_builder = StateGraph(WeatherState)
weather_builder.add_node("weather_assistant", weather_assistant)
weather_builder.add_node("weather_formatter", weather_formatter)
weather_builder.add_edge(START, "weather_assistant")
weather_builder.add_edge("weather_assistant", "weather_formatter")
weather_builder.add_edge("weather_formatter", END)

weather_graph = weather_builder.compile()
```

### 4️⃣ Building the Restaurant Subgraph 🍽️

We use DuckDuckGo’s search tool to fetch restaurant suggestions.

```python
def get_restaurants_duckduckGo_tool(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

tools = [get_restaurants_duckduckGo_tool]
llm_with_restaurant_tool = llm.bind_tools(tools)

# Restaurant assistant node
def restaurant_assistant(state: RestaurantState):
    result = llm_with_restaurant_tool.invoke(state["messages"])
    return {"messages": [result]}

def restaurants_list_formatter(state: RestaurantState):
    search_result = state["messages"][-1].content
    structured_llm = llm.with_structured_output(Restaurant)
    restaurants = structured_llm.invoke(search_result)
    return {"restaurants": restaurants}

# Building the restaurant graph
restaurant_builder = StateGraph(RestaurantState)
restaurant_builder.add_node("restaurant_assistant", restaurant_assistant)
restaurant_builder.add_node("restaurants_list", restaurants_list_formatter)
restaurant_builder.add_edge(START, "restaurant_assistant")
restaurant_builder.add_edge("restaurant_assistant", "restaurants_list")
restaurant_builder.add_edge("restaurants_list", END)

restaurant_graph = restaurant_builder.compile()
```

### 5️⃣ Combining Subgraphs into a Parent Graph 🌐

The parent graph calls both subgraphs and provides the final recommendation.

```python
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]
    location: str
    date: str
    event: str
    restaurants: list[Restaurant]
    weather_report: str
    recommendation: str

def query_analyzer(state: ParentState):
    return {
        "location": "Bangalore",
        "date": "Sunday",
        "event": "Date night",
        "messages": state['messages']
    }

def recommendation_analyzer(state: ParentState):
    summary_prompt = (
        f"Weather report: {state['weather_report']}\n"
        f"Restaurants: {state['restaurants']}\n"
        "Based on this, suggest if going out is a good idea or if staying in is better."
    )
    result = llm.invoke(summary_prompt)
    return {"recommendation": result.content}

# Building the parent graph
entry_builder = StateGraph(ParentState)
entry_builder.add_node("query_analyzer", query_analyzer)
entry_builder.add_node("weather_graph", weather_graph)
entry_builder.add_node("restaurant_graph", restaurant_graph)
entry_builder.add_node("recommendation_analyzer", recommendation_analyzer)

entry_builder.add_edge(START, "query_analyzer")
entry_builder.add_edge("query_analyzer", "weather_graph")
entry_builder.add_edge("query_analyzer", "restaurant_graph")
entry_builder.add_edge("weather_graph", "recommendation_analyzer")
entry_builder.add_edge("restaurant_graph", "recommendation_analyzer")
entry_builder.add_edge("recommendation_analyzer", END)

graph = entry_builder.compile()

# Invoking the graph
result = graph.invoke({
    "messages": ["Tell me about the best restaurants for date night in Bangalore for next Sunday"]
})

print(result['recommendation'])
```

### 6️⃣ Output 📝

The output from the AI recommendation system might look like this:

```
The weather on Sunday looks clear and pleasant — perfect for a night out! 🌙 Here are some top-rated restaurants in Bangalore for a date night:
1️⃣ Grasshopper — Beautiful garden seating and gourmet menu.
2️⃣ Olive Beach — Romantic ambiance with Mediterranean cuisine.
3️⃣ Byg Brewski Brewing Co. — Great vibes, craft beer, and outdoor seating.

Enjoy your evening! 💕
```

### 7️⃣ Key Takeaways 🔑

- **Subgraph Power:** We used subgraphs to modularize the workflow for weather and restaurant suggestions.
- **Tool Integration:** DuckDuckGo and weather APIs brought real-world data into our system.
- **Multi-Agent System:** The weather and restaurant subgraphs worked independently, and their results were combined seamlessly in the parent graph.
- **State Management:** Data flowed efficiently between subgraphs and the parent graph.

---

## ✅ You Made It!

Congratulations on reaching the end of this series! You’ve built a fully functional, multi-agent AI system using LangGraph. Keep experimenting, refining, and building smarter agents. This is just the beginning of what’s possible with LangGraph!