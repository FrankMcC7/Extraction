import os
import pandas as pd
import fitz  # PyMuPDF
import camelot
import pdfplumber
import numpy as np

# Function to extract text using PyMuPDF
def extract_text_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract tables using pdfplumber
def extract_tables_with_pdfplumber(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if table:  # Ensure table is not None
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    return tables

# Function to extract tables using Camelot
def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    tables = [table.df for table in tables]  # Convert to DataFrame
    return tables

# Function to evaluate the quality of extracted tables using heuristics
def evaluate_extraction(tables):
    scores = []
    for table in tables:
        if not table.empty:
            # Simple heuristic: score based on number of rows and columns
            score = table.shape[0] * table.shape[1]
            scores.append(score)
        else:
            scores.append(0)
    return np.mean(scores) if scores else 0

# Function to clean and format the extracted tables
def clean_and_format_tables(tables):
    cleaned_tables = []
    for table in tables:
        if table is None or table.empty:
            continue
        # Drop completely empty rows and columns
        table.dropna(how='all', inplace=True)
        table.dropna(how='all', axis=1, inplace=True)
        
        # Fill merged cells if necessary (example: forward fill)
        table.ffill(inplace=True)
        
        # Remove special characters and whitespace from column names
        table.columns = [str(col).strip().replace('\n', ' ').replace('\r', ' ') for col in table.columns]
        
        # Convert data types if necessary (example: convert numeric columns)
        for col in table.columns:
            if isinstance(table[col], pd.Series):
                try:
                    table[col] = pd.to_numeric(table[col])
                except ValueError:
                    print(f"Could not convert column {col} to numeric")
                
        cleaned_tables.append(table)
    return cleaned_tables

# Function to save extracted tables to Excel
def save_tables_to_excel(tables, output_path):
    if not tables:
        raise RuntimeError("No tables to save.")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, table in enumerate(tables):
                sheet_name = f"Table_{i+1}"[:31]  # Excel sheet names must be 31 characters or less
                table.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving tables to Excel: {e}")

# Main function to orchestrate the process
def main(pdf_path, output_excel_path):
    # Extract tables using different methods
    tables_pdfplumber = extract_tables_with_pdfplumber(pdf_path)
    tables_camelot = extract_tables_with_camelot(pdf_path)

    # Evaluate the quality of each extraction method using heuristics
    score_pdfplumber = evaluate_extraction(tables_pdfplumber)
    score_camelot = evaluate_extraction(tables_camelot)

    # Choose the best extraction method based on heuristic score
    best_method = max([(tables_pdfplumber, score_pdfplumber), 
                       (tables_camelot, score_camelot)], key=lambda x: x[1])

    best_tables = best_method[0]

    # Clean and format tables
    cleaned_tables = clean_and_format_tables(best_tables)

    # Save cleaned tables to Excel
    save_tables_to_excel(cleaned_tables, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/Asian-Paints-27-10-2023-khan.pdf'
    output_excel_path = '/mnt/data/labeled_extracted_tables.xlsx'
    main(pdf_path, output_excel_path)
