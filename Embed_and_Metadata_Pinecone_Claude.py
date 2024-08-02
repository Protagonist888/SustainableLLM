# Embed_and_Metadata_Pinecone_Claude
# This script extracts, cleans, tokenizes, and transforms text from PDFs
# and creates embeddings and extracts metadata to a JSON output formatted for pinecone

# Concluded and processed 684 chunks from PDFs.

import os
import json
import PyPDF2
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from datetime import datetime

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize BERT tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def tokenize_and_lemmatize(text: str) -> List[str]:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]

def create_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def extract_metadata(file_path: str) -> Dict[str, Any]:
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

def process_pdf(file_path: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
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
                "chunk_text": chunk_text[:1000]
            }
        }
        results.append(vector_object)
    
    return results

def process_pdf_folder(folder_path: str) -> List[Dict[str, Any]]:
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
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Main execution
if __name__ == "__main__":
    pdf_folder = "C:/Users/mchung/Hydrogen Research/Project Information/Mission Innovation/Data/Berryesa Full Case Study"
    output_json = "pinecone_vectors.json"
    
    processed_data = process_pdf_folder(pdf_folder)
    save_to_json(processed_data, output_json)
    
    print(f"Processed {len(processed_data)} chunks from PDFs. Results saved to {output_json}")