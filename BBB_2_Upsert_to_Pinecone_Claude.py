# Upsert JSON vectors to Pinecone from Claude
# WARNING: SSL verification is disabled. This is NOT SECURE and should only be used for testing.
# Many vectors failed. I hypothesize these are images, but need to double check.

import json
import os
import time
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
from pinecone import Pinecone, PineconeException
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BATCH_SIZE = 100
NAMESPACE_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def load_environment() -> tuple:
    """Load environment variables."""
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set in .env file")
    
    return api_key, index_name

def initialize_pinecone(api_key: str, index_name: str) -> Pinecone:
    """Initialize Pinecone client and connect to index with SSL verification disabled."""
    try:
        pc = Pinecone(api_key=api_key, ssl_verify=False)
        index = pc.Index(index_name)
        return index
    except PineconeException as e:
        raise ConnectionError(f"Failed to initialize Pinecone: {e}")

def adjust_vector_dimension(vector: List[float], target_dim: int) -> List[float]:
    """Adjust the dimension of the vector to match the target dimension."""
    current_dim = len(vector)
    if current_dim == target_dim:
        return vector
    elif current_dim > target_dim:
        return vector[:target_dim]  # Truncate
    else:
        return vector + [0.0] * (target_dim - current_dim)  # Pad with zeros

def upsert_batch(index: Pinecone, vectors: List[Dict[str, Any]], namespace: str) -> int:
    """Upsert a batch of vectors with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            # Get the index dimension
            index_stats = index.describe_index_stats()
            index_dim = index_stats.dimension

            # Adjust vector dimensions
            adjusted_vectors = []
            for vec in vectors:
                adjusted_vec = vec.copy()
                adjusted_vec['values'] = adjust_vector_dimension(vec['values'], index_dim)
                adjusted_vectors.append(adjusted_vec)

            response = index.upsert(vectors=adjusted_vectors, namespace=namespace)
            logging.info(f"Upserted batch of {len(vectors)} vectors to namespace: {namespace}")
            return response.upserted_count
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"Error upserting batch (attempt {attempt + 1}): {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"Failed to upsert batch after {MAX_RETRIES} attempts: {e}")
                raise
    return 0  # If all retries fail

def process_json_file(file_path: str, index: Pinecone) -> None:
    # Print initial index stats
    initial_stats = index.describe_index_stats()
    logging.info(f"Initial index stats: {initial_stats}")

    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")

    # If data is a dictionary, wrap it in a list
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"Expected a dictionary or a list of items, but got {type(data)}")

    total_upserted, total_expected, last_namespace = process_data(index, data)

    logging.info(f"Processed {len(data)} items in total.")
    logging.info(f"Expected to upsert {total_expected} vectors.")
    logging.info(f"Actually upserted {total_upserted} vectors.")

    if total_upserted == total_expected:
        logging.info("All vectors were successfully upserted.")
    else:
        logging.warning(f"Warning: {total_expected - total_upserted} vectors failed to upsert.")

    # Print final index stats
    final_stats = index.describe_index_stats()
    logging.info(f"Final index stats: {final_stats}")

    # Verify all vectors
    verify_all_vectors(index, data, last_namespace)

def process_data(index: Pinecone, data: List[Dict[str, Any]]) -> tuple:
    batch = []
    namespace_counter = 0
    namespace = f"namespace_{namespace_counter}"
    total_upserted = 0
    total_expected = 0

    for i, item in enumerate(data):
        batch.append(item)
        
        if len(batch) == BATCH_SIZE or i == len(data) - 1:
            upserted_count = upsert_batch(index, batch, namespace)
            total_upserted += upserted_count
            total_expected += len(batch)
            batch = []

        if (i + 1) % NAMESPACE_SIZE == 0:
            namespace_counter += 1
            namespace = f"namespace_{namespace_counter}"

    return total_upserted, total_expected, namespace

def verify_all_vectors(index: Pinecone, data: List[Dict[str, Any]], namespace: str):
    """Verify all upserted vectors."""
    logging.info("Starting verification of all vectors...")
    time.sleep(5)  # Add a 5-second delay before verification

    total_vectors = len(data)
    verified_count = 0
    failed_count = 0

    for item in data:
        vector_id = str(item.get("id", f"auto_id_{data.index(item)}"))
        try:
            response = index.fetch(ids=[vector_id], namespace=namespace)
            if vector_id in response['vectors']:
                verified_count += 1
                logging.debug(f"Successfully verified vector {vector_id} in namespace {namespace}")
            else:
                failed_count += 1
                logging.warning(f"Failed to verify vector {vector_id} in namespace {namespace}: Not found in index")
                logging.debug(f"Vector data: {item}")
        except Exception as e:
            failed_count += 1
            logging.error(f"Error verifying vector {vector_id} in namespace {namespace}: {e}")

    logging.info(f"Verification complete. Successfully verified: {verified_count}/{total_vectors}")
    if failed_count > 0:
        logging.warning(f"Failed to verify {failed_count} vectors.")
    else:
        logging.info("All vectors were successfully verified.")

def main():
    try:
        logging.warning("WARNING: SSL verification is disabled. This is not secure and should only be used for testing.")
        api_key, index_name = load_environment()
        index = initialize_pinecone(api_key, index_name)
        
        file_path = "C:/Users/mchung/Personal/SustainabilityScoring/Sustainability/pinecone_vectors.json"
        process_json_file(file_path, index)
        
        logging.info("Upsert process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()