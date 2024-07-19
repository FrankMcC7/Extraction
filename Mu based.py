import os
import pandas as pd
import fitz  # PyMuPDF
import camelot
import pdfplumber
import tabula
import numpy as np
import spacy
from openpyxl import Workbook
from openpyxl.comments import Comment

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract tables using pdfplumber
def extract_tables_with_pdfplumber(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if table:  # Ensure table is not None
                    df = pd.DataFrame(table[1:], columns=table[0])
                    bbox = page.find_tables()[0].bbox if page.find_tables() else None
                    tables.append((page_num, bbox, df))
    return tables

# Function to extract tables using Camelot
def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    tables = [(int(table.page), table._bbox, table.df) for table in tables]  # Convert to DataFrame
    return tables

# Function to extract tables using Tabula
def extract_tables_with_tabula(pdf_path):
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, pandas_options={'header': None})
    tables = [(i, None, table) for i, table in enumerate(tables, start=1)]
    return tables

# Function to evaluate the quality of extracted tables using heuristics
def evaluate_extraction(tables):
    scores = []
    for _, _, table in tables:
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
    for _, _, table in tables:
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

# Function to extract text around tables using SpaCy
def extract_text_around_tables(pdf_path, tables):
    texts = []
    doc = fitz.open(pdf_path)
    print(f"Document has {doc.page_count} pages.")  # Debug statement
    for page_num, bbox, table in tables:
        if page_num <= 0 or page_num > doc.page_count:
            print(f"Skipping invalid page number: {page_num}")  # Debug statement
            continue
        page = doc.load_page(page_num - 1)  # PyMuPDF uses 0-based indexing
        text = page.get_text("text")
        doc_nlp = nlp(text)
        # Extract sentences around the table bbox
        for sent in doc_nlp.sents:
            if bbox:
                bbox_rect = fitz.Rect(bbox)
                if page.search_for(sent.text.strip()):
                    texts.append(sent.text.strip())
    return texts

# Function to save extracted tables to Excel with annotations
def save_tables_to_excel(tables, texts, output_path):
    if not tables:
        raise RuntimeError("No tables to save.")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, (table, text) in enumerate(zip(tables, texts)):
                sheet_name = f"Table_{i+1}"[:31]  # Excel sheet names must be 31 characters or less
                table.to_excel(writer, sheet_name=sheet_name, index=False)
                # Add comments or additional sheets with context
                sheet = writer.sheets[sheet_name]
                sheet.cell(row=1, column=table.shape[1] + 2, value="Context")
                sheet.cell(row=2, column=table.shape[1] + 2, value=text)
    except Exception as e:
        raise RuntimeError(f"Error saving tables to Excel: {e}")

# Main function to orchestrate the process
def main(pdf_path, output_excel_path):
    # Extract tables using different methods
    tables_pdfplumber = extract_tables_with_pdfplumber(pdf_path)
    tables_camelot = extract_tables_with_camelot(pdf_path)
    tables_tabula = extract_tables_with_tabula(pdf_path)

    # Evaluate the quality of each extraction method using heuristics
    score_pdfplumber = evaluate_extraction(tables_pdfplumber)
    score_camelot = evaluate_extraction(tables_camelot)
    score_tabula = evaluate_extraction(tables_tabula)

    # Choose the best extraction method based on heuristic score
    best_method = max([(tables_pdfplumber, score_pdfplumber), 
                       (tables_camelot, score_camelot),
                       (tables_tabula, score_tabula)], key=lambda x: x[1])

    best_tables = best_method[0]

    # Clean and format tables
    cleaned_tables = clean_and_format_tables(best_tables)

    # Extract text around tables
    texts = extract_text_around_tables(pdf_path, best_tables)

    # Save cleaned tables to Excel with annotations
    save_tables_to_excel(cleaned_tables, texts, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/file-TZ4zE8ty41vWPu8h80zZf8Dg'  # Use the uploaded file path
    output_excel_path = 'abc.xlsx'
    main(pdf_path, output_excel_path)
