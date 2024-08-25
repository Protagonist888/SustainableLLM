# PDF Processing and Embedding Generation for Pinecone. 
# Leveraging langchain this time.
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
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_pdf_files(directory: str) -> List[str]:
    """
    Recursively find all PDF files in the given directory and its subdirectories.

    Args:
    directory (str): The path to the directory to search for PDF files.

    Returns:
    List[str]: A list of full file paths to all PDF files found.
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def extract_text_and_metadata(pdf_path: str) -> Dict:
    """
    Extract text and metadata from a PDF file using LangChain's PyPDFLoader.

    Args:
    pdf_path (str): The path to the PDF file.

    Returns:
    Dict: A dictionary containing the extracted text and metadata.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    text = " ".join([page.page_content for page in pages])
    metadata = {
        "source": pdf_path,
        "title": os.path.basename(pdf_path),
        "page_count": len(pages)
    }
    
    return {"text": text, "metadata": metadata}

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess the extracted text.

    Args:
    text (str): The raw text extracted from the PDF.

    Returns:
    str: The cleaned and preprocessed text.
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Convert to lowercase
    text = text.lower()
    # Add more preprocessing steps as needed
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split the text into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
    text (str): The preprocessed text to be chunked.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Create embeddings for the text chunks using the specified model.

    Args:
    chunks (List[str]): A list of text chunks.
    model_name (str): The name of the embedding model to use.

    Returns:
    List[List[float]]: A list of embedding vectors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()

def create_pinecone_objects(chunks: List[str], embeddings: List[List[float]], metadata: Dict) -> List[Dict]:
    """
    Create vector objects formatted for Pinecone.

    Args:
    chunks (List[str]): A list of text chunks.
    embeddings (List[List[float]]): A list of embedding vectors.
    metadata (Dict): Metadata for the PDF.

    Returns:
    List[Dict]: A list of vector objects formatted for Pinecone.
    """
    pinecone_objects = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        pinecone_objects.append({
            "id": f"{metadata['title']}_{i}",
            "values": embedding,
            "metadata": {
                **metadata,
                "chunk": chunk,
                "chunk_id": i
            }
        })
    return pinecone_objects

def process_pdfs(directory: str, output_file: str):
    """
    Process all PDFs in the given directory and save the results to a JSON file.

    Args:
    directory (str): The path to the directory containing PDFs.
    output_file (str): The path to the output JSON file.
    """
    pdf_files = get_pdf_files(directory)
    all_pinecone_objects = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        # Extract text and metadata
        extracted_data = extract_text_and_metadata(pdf_file)
        
        # Preprocess text
        preprocessed_text = preprocess_text(extracted_data["text"])
        
        # Chunk text
        chunks = chunk_text(preprocessed_text)
        
        # Create embeddings
        embeddings = create_embeddings(chunks)
        
        # Create Pinecone objects
        pinecone_objects = create_pinecone_objects(chunks, embeddings, extracted_data["metadata"])
        
        all_pinecone_objects.extend(pinecone_objects)

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(all_pinecone_objects, f)

    print(f"Processed {len(pdf_files)} PDFs. Results saved to {output_file}")

if __name__ == "__main__":
    pdf_directory = input("C:/Users/markc/PythonAI/DataforModels/BerryesaCaseStudy")
    output_file = input("C:/Users/markc/PythonAI/SustainableLLM")
    process_pdfs(pdf_directory, output_file)