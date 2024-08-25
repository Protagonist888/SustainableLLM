import os
import PyPDF2
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

def get_pdf_files(directory):
    """
    Recursively find all PDF files in the given directory and its subdirectories.

    Args:
    directory (str): The path to the directory to search for PDF files.

    Returns:
    list: A list of full file paths to all PDF files found.

    This function walks through the directory tree and collects the paths of all
    files with a '.pdf' extension (case-insensitive).
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def analyze_pdf(file_path):
    """
    Analyze a single PDF file and extract various characteristics.

    Args:
    file_path (str): The path to the PDF file to analyze.

    Returns:
    dict: A dictionary containing the analysis results with the following keys:
        - is_text_based (bool): True if the PDF contains extractable text.
        - is_scanned (bool): True if the PDF appears to be a scanned document.
        - page_count (int): The number of pages in the PDF.
        - has_images (bool): True if the PDF contains images.
        - has_tables (bool): True if the PDF contains tables.
        - estimated_word_count (int): An estimate of the total word count.

    This function performs the following analysis:
    1. Attempts to extract text to determine if the PDF is text-based.
    2. Checks for the presence of images and tables.
    3. If no text is found, it assumes the PDF is scanned and attempts OCR.
    4. Estimates the word count based on extracted text or OCR results.
    """
    result = {
        'is_text_based': False,
        'is_scanned': False,
        'page_count': 0,
        'has_images': False,
        'has_tables': False,
        'estimated_word_count': 0
    }
    
    try:
        with open(file_path, 'rb') as file:
            # Use PyPDF2 for initial text extraction and page count
            pdf_reader = PyPDF2.PdfReader(file)
            result['page_count'] = len(pdf_reader.pages)
            
            # Check if text-based by attempting to extract text from the first page
            first_page_text = pdf_reader.pages[0].extract_text().strip()
            if first_page_text:
                result['is_text_based'] = True
                result['estimated_word_count'] = len(first_page_text.split())
            
            # Use PyMuPDF (fitz) for more detailed analysis
            doc = fitz.open(file_path)
            
            for page in doc:
                # Check for images
                if page.get_images():
                    result['has_images'] = True
                # Check for tables
                if page.find_tables():
                    result['has_tables'] = True
                
            # If no text found, assume it's scanned and attempt OCR
            if not result['is_text_based']:
                result['is_scanned'] = True
                # Perform OCR on the first page
                pix = doc[0].get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    result['estimated_word_count'] = len(ocr_text.split())
        
    except Exception as e:
        print(f"Error analyzing PDF {file_path}: {str(e)}")
    
    return result

def main():
    """
    Main function to orchestrate the PDF analysis process.

    This function performs the following steps:
    1. Prompts the user for the directory containing PDF files.
    2. Finds all PDF files in the specified directory.
    3. Analyzes each PDF file.
    4. Compiles the results into a pandas DataFrame.
    5. Saves the results to a CSV file.
    6. Prints a summary of the analysis.

    The function uses tqdm to show a progress bar during the analysis process.
    """
    pdf_directory = input("Enter the directory path containing PDF files: ")
    pdf_files = get_pdf_files(pdf_directory)
    
    if not pdf_files:
        print("No PDF files found in the specified directory.")
        return
    
    results = []
    for pdf_path in tqdm(pdf_files, desc="Analyzing PDFs"):
        analysis = analyze_pdf(pdf_path)
        file_stats = os.stat(pdf_path)
        results.append({
            'name': os.path.basename(pdf_path),
            'size': file_stats.st_size,
            'created_time': file_stats.st_ctime,
            'modified_time': file_stats.st_mtime,
            **analysis
        })
    
    df = pd.DataFrame(results)
    output_file = 'pdf_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")
    
    # Provide summary statistics
    print("\nSummary:")
    print(f"Total PDFs analyzed: {len(df)}")
    print(f"Text-based PDFs: {df['is_text_based'].sum()}")
    print(f"Scanned PDFs: {df['is_scanned'].sum()}")
    print(f"PDFs with images: {df['has_images'].sum()}")
    print(f"PDFs with tables: {df['has_tables'].sum()}")
    print(f"Average estimated word count: {df['estimated_word_count'].mean():.2f}")
    print(f"Total estimated word count: {df['estimated_word_count'].sum()}")

if __name__ == "__main__":
    main()