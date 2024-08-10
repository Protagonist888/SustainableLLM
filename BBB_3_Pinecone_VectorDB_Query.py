# Pinecone_VectorDB_Query
# This script queries the VectorDB in Pinecone using a model that outputs 384-dimensional embeddings
# Must specify namespace to point to specific location
# Now supports multiple simultaneous queries

import os
import warnings
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeException
from pinecone.grpc import PineconeGRPC as PineconeGRPC
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import requests
import urllib3

# Disable GPU usage if not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
pinecone_index = None
model = None

def load_environment() -> tuple:
    """Load environment variables."""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    verify_ssl = os.getenv("VERIFY_SSL", "True").lower() == "true"
    
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set in .env file")
    
    return api_key, index_name, verify_ssl

def initialize_pinecone(api_key: str, index_name: str, verify_ssl: bool) -> None:
    """Initialize Pinecone client and connect to index."""
    global pinecone_index
    try:
        if not verify_ssl:
            warnings.warn("SSL verification is disabled. This is not secure and should not be used in production.", UserWarning)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            pc = Pinecone(api_key=api_key, ssl_verify=False)
        else:
            pc = Pinecone(api_key=api_key)
        pinecone_index = pc.Index(index_name)
        print(f"Successfully connected to Pinecone index: {index_name}")
    except PineconeException as e:
        raise ConnectionError(f"Failed to initialize Pinecone: {e}")

def initialize_embedding_model() -> None:
    """Initialize a sentence embedding model that outputs 384-dimensional embeddings."""
    global model
    # 'all-MiniLM-L6-v2' outputs 384-dimensional embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Embedding model initialized and running on: {device}")

def get_embedding(text: str) -> list:
    """Generate embedding for the given text."""
    global model
    
    if model is None:
        raise RuntimeError("Embedding model not initialized")
    
    # Generate embeddings
    embedding = model.encode(text, convert_to_tensor=True)
    
    # Move to CPU if it's on GPU, then convert to list
    return embedding.cpu().numpy().tolist()

def query_pinecone(query_text: str, top_k: int = 5) -> dict:
    """Query Pinecone index with the given text."""
    global pinecone_index
    
    if pinecone_index is None:
        raise RuntimeError("Pinecone index not initialized")
    
    # Generate embedding for the query text
    query_vector = get_embedding(query_text)
    
    # Query Pinecone
    results = pinecone_index.query(namespace="namespace_0", vector=query_vector, top_k=top_k, include_metadata=True)
    
    return results

def multi_query_pinecone(query_texts: list, top_k: int = 5) -> list:
    """Query Pinecone index with multiple texts."""
    return [query_pinecone(query, top_k) for query in query_texts]

def print_results(results: dict, query: str) -> None:
    """Print the query results in a formatted manner."""
    print(f"\nTop results for query: '{query}'")
    for match in results['matches']:
        print(f"ID: {match['id']}")
        print(f"Score: {match['score']:.4f}")
        print(f"Metadata: {match['metadata']}\n")

# Example usage
if __name__ == "__main__":
    try:
        api_key, index_name, verify_ssl = load_environment()
        initialize_pinecone(api_key, index_name, verify_ssl)
        
        print("Initializing embedding model...")
        initialize_embedding_model()
        
        print("Querying Pinecone...")
        
        # Example with multiple text inputs
        text_queries = [
            "Who is Anthony Kane?",
            "What is Berryessa Transit Center?",
            "Tell me about sustainable transit"
        ]
        results = multi_query_pinecone(text_queries)
        
        for query, result in zip(text_queries, results):
            print_results(result, query)
        
        # Print index stats
        stats = pinecone_index.describe_index_stats()
        print(f"\nIndex Stats:")
        print(f"Dimension: {stats['dimension']}")
        print(f"Index Fullness: {stats['index_fullness']:.2%}")
        print(f"Total Vector Count: {stats['total_vector_count']}")

    except requests.exceptions.SSLError as e:
        print(f"SSL Error occurred: {e}")
        print("\nTo temporarily bypass SSL verification (not recommended for production):")
        print("1. Set VERIFY_SSL=False in your .env file")
        print("2. Re-run the script")
        print("\nFor a more secure solution, consider updating your SSL certificates or consulting your network administrator.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()