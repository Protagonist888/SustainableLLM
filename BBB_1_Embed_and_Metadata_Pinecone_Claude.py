# PDF Processing and Embedding Generation for Pinecone
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
import PyPDF2
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the all-MiniLM-L6-v2 model for creating 384-dimensional embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text content from a PDF file.
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_text(text: str) -> str:
    """
    Removes punctuation and extra whitespace from the text.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    Tokenizes the text, removes stopwords, and lemmatizes the tokens.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]

def create_embedding(text: str) -> List[float]:
    """
    Creates a 384-dimensional embedding for the given text using all-MiniLM-L6-v2.
    """
    return model.encode(text).tolist()

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts metadata from a PDF file, including filename, size, creation date, etc.
    """
    metadata = {}
    metadata['filename'] = os.path.basename(file_path)
    metadata['file_size'] = os.path.getsize(file_path)
    metadata['creation_date'] = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    metadata['last_modified'] = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        metadata['num_pages'] = len(reader.pages)
        
        if reader.metadata:
            metadata['author'] = reader.metadata.get('/Author', 'Unknown')
            metadata['creation_date_pdf'] = reader.metadata.get('/CreationDate', 'Unknown')
    
    return metadata

def process_pdf(file_path: str, chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    Processes a single PDF file:
    - Extracts and cleans text
    - Chunks the text
    - Creates embeddings for each chunk
    - Extracts metadata
    - Returns a list of vector objects ready for Pinecone
    """
    metadata = extract_metadata(file_path)
    raw_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    tokens = tokenize_and_lemmatize(cleaned_text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    results = []
    for i, chunk in enumerate(chunks):
        chunk_text = ' '.join(chunk)
        embedding = create_embedding(chunk_text)
        
        vector_object = {
            "id": f"{metadata['filename']}_chunk_{i}",
            "values": embedding,
            "metadata": {
                **metadata,
                "chunk_id": i,
                "chunk_text": chunk_text[:1000]  # Truncate chunk text to 1000 characters
            }
        }
        results.append(vector_object)
    
    return results

def process_pdf_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Processes all PDF files in the specified folder.
    """
    results = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(folder_path, filename)
        try:
            pdf_results = process_pdf(file_path)
            results.extend(pdf_results)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return results

def save_to_json(data: List[Dict[str, Any]], output_file: str):
    """
    Saves the processed data to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Main execution
if __name__ == "__main__":
    pdf_folder = "C:/Users/mchung/Hydrogen Research/Project Information/Mission Innovation/Data/Berryesa Full Case Study"
    output_json = "BBB_pinecone_vectors.json"
    
    processed_data = process_pdf_folder(pdf_folder)
    save_to_json(processed_data, output_json)
    
    print(f"Processed {len(processed_data)} chunks from PDFs. Results saved to {output_json}")