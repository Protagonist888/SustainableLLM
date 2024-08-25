# PDF Processing and Embedding Generation for Pinecone. 
# Leveraging langchain this time. Using "myenv" venv
#
# This script processes PDF documents to prepare data for upload to Pinecone, a vector database.
# It performs the following steps:
# 1. Extracts text from PDF files
# 2. Cleans and preprocesses the text
# 3. Creates embeddings using the all-MiniLM-L6-v2 model (384 dimensions)
# 4. Extracts metadata from PDFs
# 5. Chunks the text and creates vector objects for each chunk
# 6. Saves the processed data to a JSON file formatted for Pinecone
#
# The script processes all PDFs in a specified folder and outputs a single JSON file
# containing vector objects ready for upload to Pinecone.
import os
import json
import torch
import fitz  # PyMuPDF
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ... (previous functions remain the same)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split the text into chunks.

    Args:
    text (str): The preprocessed text to be chunked.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

        # Break the loop if we've reached the end of the text
        if end == text_length:
            break

    return chunks

# ... (rest of the code remains the same)