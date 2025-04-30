# LangChain Tutorial

This tutorial provides a comprehensive guide to working with LangChain, a framework for developing applications powered by language models. We'll cover the basics of LangChain, including prompt templates, memory, tools, and chains.

## Table of Contents
- [Installation](#installation)
- [Basic LLM Calling](#basic-llm-calling)
- [Prompt Templates](#prompt-templates)
- [Memory](#memory)
- [Tools](#tools)
- [Chains](#chains)

## Installation

Before you begin, you need to install the required packages:

```python
pip install langchain openai python-dotenv
pip install -U langchain langchain-openai
```

## Basic LLM Calling

The first step is to set up your environment and make a basic call to a language model:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")
HOST = os.getenv("HOST")
MODEL = os.getenv("MODEL")

# Initialize LangChain OpenAI client (Databricks API compatible)
chat = ChatOpenAI(
    openai_api_key=TOKEN,
    openai_api_base=HOST,
    model_name=MODEL
)

# Create a message
messages = [HumanMessage(content="Hello! How are you?")]

# Get response (use .invoke() instead of direct call)
response = chat.invoke(messages)

# Print the response
print(response.content)
```

## Prompt Templates

LangChain provides tools for creating reusable prompt templates:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")
HOST = os.getenv("HOST")
MODEL = os.getenv("MODEL")

# Initialize LangChain OpenAI client (Databricks API compatible)
chat = ChatOpenAI(
    openai_api_key=TOKEN,
    openai_api_base=HOST,
    model_name=MODEL
)

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful AI assistant. Answer the following: {question}"
)

# Define a Runnable chain
chain = prompt | chat

# Run the chain using invoke()
response = chain.invoke({"question": "What is LangChain?"})

# Print response
print(response.content)
```

## Memory

LangChain allows you to create conversational agents with memory:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")
HOST = os.getenv("HOST")
MODEL = os.getenv("MODEL")

# Initialize Databricks OpenAI-compatible LLM
llm = ChatOpenAI(
    openai_api_key=TOKEN,
    openai_api_base=HOST,
    model_name=MODEL,
    streaming=True,  # Enable streaming
    callbacks=[StreamingStdOutCallbackHandler()]  # Prints tokens as they arrive
)

# Define a chat prompt with memory
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful AI assistant providing answers to user queries."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Memory Configuration
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the chain
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# 🧠 First Interaction (Stores conversation in memory)
response1 = conversation({"question": "Who won the FIFA World Cup 2022?"})
print("\nFirst Response:", response1["text"])

# 🧠 Second Interaction (Remembers previous conversation)
response2 = conversation({"question": "Who was the captain of the winning team?"})
print("\nSecond Response:", response2["text"])
```

## Tools

LangChain integrates with various tools that can be used to enhance your applications:

```python
# Install the Wikipedia tool
pip install wikipedia

# Use the Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

print(tool.name)
print(tool.description)
print(tool.args)

tool.run({"query": "deep seek"})
```

## Chains

LangChain allows you to create chains of operations:

### Basic Chain

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")
HOST = os.getenv("HOST")
MODEL = os.getenv("MODEL")

# Initialize Databricks OpenAI-compatible LLM
llm = ChatOpenAI(
    openai_api_key=TOKEN,
    openai_api_base=HOST,
    model_name=MODEL,
)

# Define a single chain
chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Suggest a company name for {product}.")
)

# Run the chain
result = chain.run({"product": "gaming laptop"})
print(result)
```

### Sequential Chain

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")
HOST = os.getenv("HOST")
MODEL = os.getenv("MODEL")

# Initialize Databricks OpenAI-compatible LLM
llm = ChatOpenAI(
    openai_api_key=TOKEN,
    openai_api_base=HOST,
    model_name=MODEL,
)

# Define chains
chain_one = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Give me just one word output. I need a company name for {product}."),
    output_key="company_name"
)

chain_two = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("Describe {company_name} in one sentence."),
    output_key="description"
)

# Create a simple SequentialChain
simple_chain = SequentialChain(
    chains=[chain_one, chain_two],
    input_variables=["product"],
    output_variables=["company_name", "description"]
)

# Run the chain
result = simple_chain({"product": "gaming laptop"})
print(result)
```

### Detailed Sequential Chain Example

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")
HOST = os.getenv("HOST")
MODEL = os.getenv("MODEL")

# Initialize Databricks OpenAI-compatible LLM
llm = ChatOpenAI(
    openai_api_key=TOKEN,
    openai_api_base=HOST,
    model_name=MODEL,
)

# Define the first prompt (generates company name)
first_prompt = ChatPromptTemplate.from_template(
    """
    What is the best name to describe a company that makes {product}? 
    Just me just one single company name in plain text in one word only.
    No explanation, title, headings etc. THe word should explain the company's product.
    """
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="company_name")

# Define the second prompt (generates description)
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20-word description for the following company: {company_name}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="description")

# Create a SequentialChain
overall_simple_chain = SequentialChain(
    chains=[chain_one, chain_two],
    input_variables=["product"],  # Start with product
    output_variables=["company_name", "description"],  # Expected outputs
    verbose=True
)

# Run the chain
response = overall_simple_chain({"product": "gaming laptop"})
print(response)
```

## Conclusion

This tutorial covered the essentials of using LangChain, including making basic LLM calls, working with prompt templates, implementing conversational memory, using tools, and creating chains. LangChain provides a powerful framework for building complex applications powered by language models, making it easier to create chatbots, question-answering systems, and other AI-powered tools. 