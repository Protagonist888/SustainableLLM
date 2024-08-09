# Pinecone_VectorDB_Query
# This script queries the VectorDB in Pinecone using a lightweight model optimized for CPU

# Required imports
import os
import warnings
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeException
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import urllib3

# Disable GPU usage for TensorFlow (used by sentence-transformers)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    except PineconeException as e:
        raise ConnectionError(f"Failed to initialize Pinecone: {e}")

def initialize_embedding_model() -> None:
    """Initialize a lightweight sentence embedding model optimized for CPU."""
    global model
    # 'paraphrase-MiniLM-L3-v2' is a lightweight model that performs well on CPUs
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def query_pinecone(query_text: str, top_k: int = 5) -> dict:
    """Query Pinecone index with the given text."""
    global pinecone_index, model
    
    if not pinecone_index or model is None:
        raise RuntimeError("Pinecone index or embedding model not initialized")
    
    # Generate embedding for the query text
    query_vector = model.encode(query_text, show_progress_bar=False).tolist()
    
    # Query Pinecone
    results = pinecone_index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    return results

# Example usage
if __name__ == "__main__":
    try:
        api_key, index_name, verify_ssl = load_environment()
        initialize_pinecone(api_key, index_name, verify_ssl)
        
        print("Initializing embedding model...")
        initialize_embedding_model()
        
        print("Querying Pinecone...")
        
        # Example with text input
        text_query = "What is the Envision metric QL1.2?"
        results = query_pinecone(text_query)
        
        print("\nTop results:")
        for match in results['matches']:
            print(f"ID: {match['id']}, Score: {match['score']}")
            print(f"Metadata: {match['metadata']}\n")
    
    except requests.exceptions.SSLError as e:
        print(f"SSL Error occurred: {e}")
        print("\nTo temporarily bypass SSL verification (not recommended for production):")
        print("1. Set VERIFY_SSL=False in your .env file")
        print("2. Re-run the script")
        print("\nFor a more secure solution, consider updating your SSL certificates or consulting your network administrator.")
    except Exception as e:
        print(f"An error occurred: {e}")