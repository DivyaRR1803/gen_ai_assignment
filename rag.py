import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import asyncio
import os
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


# Initialize the model and ChromaDB client
model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cpu')
persistent_storage_path = "./chromadb_storage"  
# Create the directory if it does not exist
if not os.path.exists(persistent_storage_path):
    os.makedirs(persistent_storage_path)
    

# Initialize ChromaDB client with a persistent storage path
chroma_client = chromadb.PersistentClient(
    path=persistent_storage_path,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

collection = chroma_client.get_or_create_collection("document_embeddings")

# Function to perform similarity search
def similarity_search(query, top_n=2):
    # Embed the query
    query_embedding = model.encode([query])[0].tolist()  # Encoding the query and converting to list
    
    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    
    # Check if results exist and contain the expected structure
    if 'results' not in results or not results['results']:
        print("No results found in ChromaDB for the query.")
        return []

    # Format the results for readability
    similarity_scores = [
        {
            'chunk': result['document'],
            'similarity_score': result['distance'],
            'title': result['metadata'].get('title', 'No title'),
            'doc_filename': result['metadata'].get('doc_filename', 'No filename')
        }
        for result in results['results'][0]
    ]

    # Return the top_n most relevant chunks
    return similarity_scores


# Example usage
async def query_similarity_search(query):
    top_n = 2  # Number of top similar chunks to retrieve
    result = similarity_search(query, top_n=top_n)
    
    print(f"Top {top_n} similar chunks for the query:")
    for idx, item in enumerate(result):
        print(f"\nRank {idx+1}:")
        print(f"Title: {item['title']}")
        print(f"Document Filename: {item['doc_filename']}")
        print(f"Similarity Score: {item['similarity_score']}")
        print(f"Chunk: {item['chunk']}")

# Example query
query = "what is kubernetes"
asyncio.run(query_similarity_search(query))
