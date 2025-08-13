# Retrieval Augmented Generation (RAG) Tutorial

This tutorial provides a comprehensive guide to implementing Retrieval Augmented Generation (RAG), a technique that enhances large language models by retrieving relevant information from external sources before generating responses.

## Table of Contents
- [Introduction to RAG](#introduction-to-rag)
- [Setting Up the Environment](#setting-up-the-environment)
- [Basic LLM Calling](#basic-llm-calling)
- [Implementing RAG Components](#implementing-rag-components)
  - [Vector Database Setup](#vector-database-setup)
  - [Query Retrieval](#query-retrieval)
- [Enhancing RAG with Query Refinement](#enhancing-rag-with-query-refinement)
- [Building the Complete RAG Pipeline](#building-the-complete-rag-pipeline)

## Introduction to RAG

Retrieval Augmented Generation (RAG) combines the knowledge retrieved from an external database with the capabilities of large language models to produce more accurate, factual, and contextually relevant responses. It's particularly effective for answering questions about specific documents, company information, or any domain-specific content.

The key components of a RAG system include:
1. A vector database to store embedded documents
2. A retrieval mechanism to find relevant content
3. An LLM to generate responses based on the retrieved content

## Setting Up the Environment

First, let's set up our environment by importing the necessary libraries and initializing the OpenAI client:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get('TOKEN')
HOST = os.environ.get('HOST')
MODEL = os.environ.get('MODEL')

llm_client = OpenAI(
  api_key=TOKEN,
  base_url=HOST
)
```

## Basic LLM Calling

Before diving into RAG, let's verify that our basic LLM call works:

```python
chat_completion = llm_client.chat.completions.create(
  messages=[
    {"role": "system", "content": "You are an AI assistant"},
    {"role": "user", "content": "Tell me about Large Language Models"}
  ],
  model=MODEL,
  max_tokens=256
)

print(chat_completion.choices[0].message.content)
```

## Implementing RAG Components

### Vector Database Setup

For RAG, we need a vector database to store and search through embedded documents. Here, we'll use ChromaDB and SentenceTransformers for embedding:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# Load embedding and cross-encoder models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="knowledge_base")
collection = client.get_or_create_collection(name="gen_ai_crash_course_collection")
```

### Query Retrieval

Next, we'll implement functions to retrieve relevant content from our vector database:

```python
def fetch_top_relevant_queries(query, collection, top_k=10):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0], results['metadatas'][0], results['distances'][0]

def retrieve_relevant_content(query):
    documents, metadata, scores = fetch_top_relevant_queries(query, collection, top_k=10)
    # Compute cross-encoder scores
    scored_queries = []
    for doc, meta, score in zip(documents, metadata, scores):
        doc_text = doc.replace("\n", " ")
        cross_score = cross_encoder.predict([(query, doc_text)])[0]
        scored_queries.append({
            "document": doc_text, 
            "metadata": meta, 
            "similarity_score": score, 
            "cross_encoder_score": cross_score
        })

    # Sort by cross-encoder score and take top 3
    top_docs = sorted(scored_queries, key=lambda x: x['cross_encoder_score'], reverse=True)[:3]

    # Combine top 3 documents into a single paragraph
    combined_text = " ".join([item['document'] for item in top_docs])

    return combined_text
```

This retrieval function:
1. Embeds the user query
2. Searches for similar documents in the vector database
3. Re-ranks the results using a cross-encoder model
4. Returns the top 3 most relevant documents combined into a single text

## Enhancing RAG with Query Refinement

A common challenge with RAG is that user queries might not match the exact phrasing in our vector database. We can enhance retrieval by first refining the user's query:

```python
def enhance_prompt(query):
    """Enhances the user query to improve retrieval accuracy."""
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant for L&T group. Your task is to refine user queries for better information retrieval. Make the query clearer, more specific, and structured to help retrieve relevant documents accurately."},
            {"role": "user", "content": f"Enhance this query for better retrieval: {query}"}
        ],
        model=MODEL,
        max_tokens=50
    )
    return chat_completion.choices[0].message.content.strip()
```

## Building the Complete RAG Pipeline

Now, let's combine all components into a complete RAG pipeline:

```python
def enhanced_llm_call(user_query):
    """Enhances query, retrieves relevant content, and gets final response."""
    
    # Step 1: Enhance the query for better retrieval
    refined_query = enhance_prompt(user_query)

    # Step 2: Retrieve relevant content using the enhanced query
    relevant_content = retrieve_relevant_content(refined_query)

    # Step 3: Send both the retrieved content and the original user question to LLM
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant. Answer the user's question using the given context."},
            {"role": "user", "content": f"Context:\n{relevant_content}\n\nUser Question: {user_query}"}
        ],
        model=MODEL,
        max_tokens=512
    )

    return chat_completion.choices[0].message.content

# Example Usage
user_query = "Who is your chairman"
final_output = enhanced_llm_call(user_query)
print(final_output)
```

This complete RAG pipeline:
1. Refines the user's query for better retrieval
2. Retrieves relevant content from the vector database
3. Feeds both the context and the original question to the LLM
4. Returns a response that's grounded in the retrieved facts

## Conclusion

RAG is a powerful technique that combines the knowledge retrieval capabilities of vector databases with the language generation abilities of LLMs. This approach results in more accurate, factual, and contextually relevant responses, especially when dealing with domain-specific information.

Key benefits of RAG include:
- Reduced hallucinations since responses are grounded in retrieved facts
- The ability to access information beyond the LLM's training data
- Up-to-date responses as the vector database can be continuously updated
- Domain-specific expertise without fine-tuning the base model

By implementing the RAG pipeline as described in this tutorial, you can create AI assistants that provide more accurate and helpful responses to user queries about specific content domains. 