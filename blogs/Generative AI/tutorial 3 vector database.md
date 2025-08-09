# Vector Database Tutorial

This tutorial provides a comprehensive guide to working with vector databases for AI applications. We'll explore how to process text data, create embeddings, store them in a vector database, and perform semantic search and retrieval.

## Table of Contents
- [Reading and Processing Data](#reading-and-processing-data)
- [Chunking Data](#chunking-data)
- [Creating Embeddings](#creating-embeddings)
- [Storing Embeddings in a Collection](#storing-embeddings-in-a-collection)
- [Querying Vector Database](#querying-vector-database)
- [Using OpenAI with System and User Prompts](#using-openai-with-system-and-user-prompts)
- [Creating Augmented Chunks](#creating-augmented-chunks)
- [Performing Re-Ranking](#performing-re-ranking)

## Reading and Processing Data

First, we need to extract content from various sources like websites and PDFs, then save it as text files for further processing.

```python
import os
import re
import json
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Combined dictionary mapping URLs to filenames (both webpages and PDFs)
url_filename_mapping = {
    "https://www.larsentoubro.com/corporate/about-lt-group/overview/": "Larsen_Toubro_Overview.txt",
    "https://www.larsentoubro.com/corporate/about-lt-group/technology-for-growth/": "Larsen_Toubro_Technology_for_Growth.txt",
    # Add more URLs as needed
    
    # PDF URLs
    "https://annualreview.larsentoubro.com/download/L&T-Annual-Review-2024.pdf": "LT_Annual_Review_2024.pdf",
    "https://annualreview.larsentoubro.com/download/L&T%20Annual%20Review%202023.pdf": "LT_Annual_Review_2023.pdf",
    # Add more PDF URLs as needed
}

def save_webpages_as_text(urls):
    for url, filename in urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")

            with open(f"dataset/{filename}", "w", encoding="utf-8") as file:
                file.write(text)

            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

def download_and_parse_pdfs(pdf_urls):
    for url, filename in pdf_urls.items():
        pdf_filepath = f"dataset/{filename}"
        txt_filename = filename.replace(".pdf", ".txt")
        txt_filepath = f"dataset/{txt_filename}"

        try:
            # Step 1: Download the PDF
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(pdf_filepath, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            print(f"Downloaded: {pdf_filepath}")

            # Step 2: Extract text from the PDF
            with open(pdf_filepath, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])

            # Step 3: Save extracted text to a .txt file
            with open(txt_filepath, "w", encoding="utf-8") as txt_file:
                txt_file.write(extracted_text)

            print(f"Extracted text saved: {txt_filepath}")

        except Exception as e:
            print(f"Error processing {url}: {e}")

def process_text_files(urls):
    """
    Processes all URLs, sending PDFs to download_and_parse_pdfs and webpages to save_webpages_as_text.
    """
    pdf_urls = {url: filename for url, filename in urls.items() if url.endswith('.pdf')}
    webpage_urls = {url: filename for url, filename in urls.items() if not url.endswith('.pdf')}

    # Process PDFs
    if pdf_urls:
        download_and_parse_pdfs(pdf_urls)

    # Process Webpages
    if webpage_urls:
        save_webpages_as_text(webpage_urls)

# Run the function to process all URLs (PDFs and Webpages)
process_text_files(url_filename_mapping)
```

## Chunking Data

After collecting the text data, we need to split it into smaller, manageable chunks for processing and storage.

```python
def preprocess_text(text, filename, url):
    """
    Preprocess the text by:
    1. Converting to lowercase.
    2. Removing excessive newlines (\n), keeping max 2 consecutive.
    3. Splitting into smaller chunks using Recursive Text Splitter.
    4. Storing chunks in the global list with metadata.
    """
    text = text.lower()
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit newlines to max 2 consecutive

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size as needed
        chunk_overlap=100  # Overlapping to maintain context
    )
    chunks = text_splitter.split_text(text)
    
    chunks_data = []

    # Append data to the global list
    for chunk in chunks:
        chunks_data.append({
            "chunk_data": chunk,
            "metadata": {
                "filename": filename,
                "url": url
            }
        })

    print(f"Processed and stored chunks for {filename}.")
    return chunks_data

def chunk_data(urls):
    # Global list to store all chunks data
    all_chunks_data = []
    # Process text files for chunking and JSON storage
    for url, filename in urls.items():
        if filename.endswith('.txt'):
            try:
                with open(f"dataset/{filename}", "r", encoding="utf-8") as file:
                    text = file.read()
                chunks_data = preprocess_text(text, filename, url)
                all_chunks_data += chunks_data
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Once all files are processed, save all chunks to a single JSON file
    with open("dataset/all_chunks_data.json", "w", encoding="utf-8") as json_file:
        json.dump(all_chunks_data, json_file, ensure_ascii=False, indent=4)

    print("All chunks processed and saved to 'all_chunks_data.json'.")
    
# Run the function to chunk all data
chunk_data(url_filename_mapping)
```

## Creating Embeddings

We'll use a pre-trained model from the `sentence-transformers` library to generate embeddings for our text chunks.

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
```

## Storing Embeddings in a Collection

Next, we'll store the embeddings in a ChromaDB collection for efficient similarity search.

```python
import json
from sentence_transformers import SentenceTransformer
import chromadb

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Chroma client and create a collection
client = chromadb.PersistentClient(path="../knowledge_base")

collection = client.get_or_create_collection(name="embeddings_collection")

def create_embeddings_from_json(json_filename):
    """
    Read the JSON file containing chunk data, generate embeddings using SentenceTransformer,
    and create a chroma collection.
    """
    # Load the chunk data from the JSON file
    with open(json_filename, 'r', encoding='utf-8') as file:
        chunk_data = json.load(file)

    # Prepare sentences (chunks) and metadata for embedding
    sentences = [chunk['chunk_data'] for chunk in chunk_data]
    metadatas = [chunk['metadata'] for chunk in chunk_data]

    # Generate embeddings using the SentenceTransformer model
    embeddings = model.encode(sentences)

    # Insert the embeddings and metadata into the Chroma collection
    for idx, embedding in enumerate(embeddings):
        # You can optionally store metadata alongside each embedding
        collection.add(
            ids=[str(idx)],  # Unique ID for each chunk
            embeddings=[embedding],
            metadatas=[metadatas[idx]],  # Store metadata (filename, URL)
            documents=[sentences[idx]]  # Store the sentence (chunk)
        )

    print(f"Created embeddings for {len(sentences)} chunks and added to Chroma collection.")

# Run the function to create embeddings and add to Chroma collection
create_embeddings_from_json('dataset/all_chunks_data.json')

# Check the number of documents in the collection
num_documents = len(collection.get()['documents'])
print(f"Number of documents in the collection: {num_documents}")
```

## Querying Vector Database

Now we can search the vector database for relevant information using semantic similarity.

```python
import numpy as np

def fetch_top_relevant_queries(query, collection, top_k=10):
    """
    Takes a query, encodes it using the SentenceTransformer, and fetches the top `top_k` relevant queries from the Chroma collection.
    """
    
    # Encode the query into an embedding
    query_embedding = model.encode([query])

    # Perform a similarity search in the Chroma collection
    results = collection.query(
        query_embeddings=query_embedding,  # The query embedding
        n_results=top_k  # Number of top results to return
    )

    # Process the results
    relevant_queries = []
    for result in results['documents']:
        relevant_queries.append({
            "document": result,  # The chunk or sentence text
            "metadata": results['metadatas'][results['documents'].index(result)],  # Metadata for each chunk
            "score": results['distances'][results['documents'].index(result)]  # Similarity score (distance)
        })

    return relevant_queries


query = "Who are the founders"
top_queries = fetch_top_relevant_queries(query, collection, top_k=10)

# Display the top 10 relevant queries
for i in range(len(top_queries[0]['metadata'])):
    print("Document: " + str(top_queries[0]['document'][i].replace("\n", " ")))
    print("Metadata: " + str(top_queries[0]['metadata'][i]))
    print("Similarity Score: " + str(top_queries[0]['score'][i]))
    print("\n\n")
```

## Using OpenAI with System and User Prompts

We can enhance our vector database retrieval by integrating it with OpenAI models.

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get('TOKEN')
HOST = os.environ.get('HOST')
MODEL = os.environ.get('MODEL')

client = OpenAI(
  api_key = TOKEN,
  base_url = HOST
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "Tell me about Large Language Models"
  }
  ],
  model=MODEL,
  max_tokens=256
)

print(chat_completion.choices[0].message.content)
```

## Creating Augmented Chunks

Augmented chunks add context to the original text, making it more likely to be retrieved when relevant.

```python
import random
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

random.seed(10)

# Load environment variables
load_dotenv()

TOKEN = os.environ.get('TOKEN')
HOST = os.environ.get('HOST')
MODEL = os.environ.get('MODEL')

# Initialize OpenAI client
client = OpenAI(
  api_key=TOKEN,
  base_url=f"{HOST}/serving-endpoints"
)

def get_random_chunk_and_generate_questions(json_file_path):
    """
    Selects a random chunk from the provided JSON file and sends it to OpenAI to generate a list of questions
    that can be answered using the chunk of text.
    """
    try:
        # Read the JSON file and load the chunks data
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            chunks_data = json.load(json_file)

        random_chunk = random.choice(chunks_data)
        
        random_chunk_data = random_chunk['chunk_data']

        # Prepare the prompt for OpenAI
        prompt = f"Please generate a set of questions that could be answered using this information: \n\n{random_chunk_data}\n\n"

        # Send the prompt to OpenAI
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant with the ability to generate relevant questions based on provided text. "
                            "Your task is to analyze the text and create insightful questions that can be answered using that text."
                            "return only the questions in plain text in multiple lines. no headings, no titles, nothing, no bulletpoints"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL,
        )

        # Get the response and print the generated questions
        questions = chat_completion.choices[0].message.content
        
        return random_chunk, questions

    except Exception as e:
        print(f"Error: {e}")

# Example usage
chunk, llm_output = get_random_chunk_and_generate_questions('dataset/all_chunks_data.json')


print(chunk)
print("\n\n\n")
print(llm_output)
```

Creating augmented chunks for a more effective retrieval:

```python
questions = llm_output.split("\n")

qes = questions[0]
augumented_chunk = f"{qes}{chunk['chunk_data']}"
print(augumented_chunk)

augumented_chunks = []
for question in questions:
    chunk_copy = {
        "chunk_data": f"{question}\n\n{chunk['chunk_data']}",
        "metadata": chunk['metadata']
    }
    chunk_copy['metadata']['is_augumented'] = True
    augumented_chunks.append(chunk_copy)

print("Number of augumented chunks: ", len(augumented_chunks))
```

Adding the augmented chunks to the collection:

```python
import json
from sentence_transformers import SentenceTransformer
import chromadb

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def add_augumented_chunks_to_collection(augumented_chunks):
    # Prepare sentences (chunks) and metadata for embedding
    sentences = [chunk['chunk_data'] for chunk in augumented_chunks]
    metadatas = [chunk['metadata'] for chunk in augumented_chunks]

    # Generate embeddings using the SentenceTransformer model
    embeddings = model.encode(sentences)
    last_idx = len(collection.get()['documents'])
    # Insert the embeddings and metadata into the Chroma collection
    for idx, embedding in enumerate(embeddings):
        # You can optionally store metadata alongside each embedding
        
        collection.add(
            ids=[str(idx+last_idx)],  # Unique ID for each chunk
            embeddings=[embedding],
            metadatas=[metadatas[idx]],  # Store metadata (filename, URL)
            documents=[sentences[idx]]  # Store the sentence (chunk)
        )

    print(f"Created embeddings for {len(sentences)} chunks and added to Chroma collection.")

# Run the function to create embeddings and add to Chroma collection
add_augumented_chunks_to_collection(augumented_chunks)
```

## Performing Re-Ranking

Re-ranking helps improve search relevance by refining initial retrieval results.

```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "Who is the chairman of L&T?"
top_queries = fetch_top_relevant_queries(query, collection, top_k=10)

# Display the top 10 relevant queries
for i in range(len(top_queries[0]['metadata'])):
    print("Document: " + str(top_queries[0]['document'][i].replace("\n", " ")))
    print("Metadata: " + str(top_queries[0]['metadata'][i]))
    print("Similarity Score: " + str(top_queries[0]['score'][i]))
    print("\n\n")
```

Computing cross-encoder scores:

```python
from sentence_transformers import CrossEncoder

# Load the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "Who is the chairman of L&T?"
top_queries = fetch_top_relevant_queries(query, collection, top_k=10)

# Compute cross-encoder scores
for i in range(len(top_queries[0]['metadata'])):
    document_text = str(top_queries[0]['document'][i].replace("\n", " "))
    
    # Compute cross-encoder relevance score
    score = cross_encoder.predict([(query, document_text)])
    
    print("Document: " + document_text)
    print("Similarity Score: " + str(top_queries[0]['score'][i]))
    print("Cross-Encoder Score: " + str(score[0]))  # Displaying the cross-encoder score
    print("\n\n")
```

Sorting results by cross-encoder score:

```python
from sentence_transformers import CrossEncoder

# Load the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "Who is the chairman of L&T?"
top_queries = fetch_top_relevant_queries(query, collection, top_k=10)

# Compute cross-encoder scores and store them with the query data
scored_queries = []
for i in range(len(top_queries[0]['metadata'])):
    document_text = str(top_queries[0]['document'][i].replace("\n", " "))
    
    # Compute cross-encoder relevance score
    score = cross_encoder.predict([(query, document_text)])[0]  
    
    # Append to list with all relevant data
    scored_queries.append({
        "document": document_text,
        "metadata": top_queries[0]['metadata'][i],
        "similarity_score": top_queries[0]['score'][i],
        "cross_encoder_score": score
    })

# Sort the queries based on cross-encoder score in descending order
scored_queries.sort(key=lambda x: x['cross_encoder_score'], reverse=True)

# Display sorted results
for item in scored_queries:
    print("Document: " + item["document"])
    print("Metadata: " + str(item["metadata"]))
    print("Similarity Score: " + str(item["similarity_score"]))
    print("Cross-Encoder Score: " + str(item["cross_encoder_score"]))
    print("\n\n")
```

## Conclusion

This tutorial has covered the essential steps for working with vector databases in AI applications:

1. Reading and processing data from various sources
2. Chunking text data into manageable pieces
3. Creating embeddings using pre-trained models
4. Storing embeddings in a vector database
5. Querying the vector database for relevant information
6. Integrating with OpenAI models
7. Creating augmented chunks for better retrieval
8. Performing re-ranking for improved search relevance

These techniques form the foundation for building powerful AI systems capable of semantic search and retrieval, which are essential for applications like question-answering systems, chatbots, and knowledge bases. 