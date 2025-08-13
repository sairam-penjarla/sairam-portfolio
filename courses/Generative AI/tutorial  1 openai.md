# OpenAI API Tutorial

This tutorial provides a comprehensive guide to working with the OpenAI API, demonstrating how to set up the environment, initialize the client, make API calls, and manage conversations.

## Table of Contents
- [Setting Up the Environment](#setting-up-the-environment)
- [Initializing the OpenAI Client](#initializing-the-openai-client)
- [Making API Calls](#making-api-calls)
- [Storing and Managing Conversations](#storing-and-managing-conversations)
- [Adding Contextual Information](#adding-contextual-information)

## Setting Up the Environment

Before working with the OpenAI API, you need to set up your environment with the necessary credentials.

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

The code above loads environment variables from a `.env` file. Create a `.env` file in your project directory with the following variables:

```
TOKEN=your_openai_api_key
HOST=your_api_host_url
MODEL=your_preferred_model
```

## Initializing the OpenAI Client

After setting up the environment, initialize the OpenAI client with your credentials:

```python
from openai import OpenAI

TOKEN = os.environ.get('TOKEN')
HOST = os.environ.get('HOST')
MODEL = os.environ.get('MODEL')

client = OpenAI(
    api_key = TOKEN,
    base_url = HOST
)
```

## Making API Calls

### Understanding Parameters

When making API calls to OpenAI, you can use several parameters to control the response:

- **`messages`**: A list of message objects defining the conversation.
- **`model`**: The model to use (e.g., `gpt-3.5-turbo`).
- **`max_tokens`**: Limits the number of tokens in the response.
- **`temperature`**: Controls randomness in output (lower values = more deterministic).

### Basic API Call

Here's a simple example of making an API call to the chat model:

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Hello! How are you?"},
    ],
    max_tokens=50,
    temperature=0.7
)

# Print the response
print(response.choices[0].message.content)
```

### Using System Messages

You can use system messages to set the behavior of the assistant:

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant who always talks in all caps."},
        {"role": "user", "content": "Tell me a random joke"},
    ],
    max_tokens=50,
    temperature=0.99
)

print(response.choices[0].message.content)
```

## Storing and Managing Conversations

OpenAI's API allows you to maintain conversations by including previous messages in your requests:

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the World Series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Say it again, but in all caps."},
    ],
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Managing Conversations with a Function

Here's a way to manage conversations using a Python function:

```python
conversations = [
  {
    "role": "system",
    "content": """
                You are an AI assistant who needs to answer user's 
                question using the provided relevant content
                """
  }
]

def invoke_llm(user_input):
    msg = {
        "role": "user",
        "content": f"user's question':{user_input}"
    }

    conversations.append(msg)

    chat_completion = client.chat.completions.create(
                        messages=conversations,
                        model=MODEL,
                        max_tokens=50,
                        temperature=0.7
                      )

    llm_output = chat_completion.choices[0].message.content

    conversations.append({
        "role": "assistant",
        "content": llm_output
    })

    print(llm_output)
```

Example usage:

```python
invoke_llm(user_input = "in which year is GTA 6 comming out")
invoke_llm(user_input = "Say it again in the form of a three line poem")
invoke_llm(user_input = "what was my original question?")
```

## Adding Contextual Information

You can enhance the AI's responses by providing additional contextual information:

```python
conversations = [
  {
    "role": "system",
    "content": """
                You are an AI assistant who needs to answer user's 
                question using the provided relevant content
                """
  }
]

def invoke_llm(content, user_input):
    msg = {
        "role": "user",
        "content": f"relavant content: {content} user's question':{user_input}"
    }

    conversations.append(msg)

    chat_completion = client.chat.completions.create(
                        messages=conversations,
                        model=MODEL,
                        max_tokens=256,
                        temperature=0.7
                      )

    llm_output = chat_completion.choices[0].message.content

    conversations.append({
        "role": "assistant",
        "content": llm_output
    })

    return llm_output

# Example usage
invoke_llm(
    content = """
            rockstar has released the trailer of GTA 6 on 2024 and has announced 
            that the game will out out in 2025
            """,
    user_input = "in which year is GTA 6 comming out"
)
```

This approach allows you to provide the AI with specific information to reference when answering questions, leading to more accurate and contextually relevant responses.

## Conclusion

This tutorial covered the basics of working with the OpenAI API, including setting up the environment, initializing the client, making API calls with various parameters, and managing conversations. With these fundamentals, you can build more complex applications that leverage OpenAI's powerful language models. 