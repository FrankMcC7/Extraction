import os
import pandas as pd
import fitz  # PyMuPDF
import camelot
import pdfplumber
import tabula
import numpy as np
from openpyxl import Workbook
from openpyxl.comments import Comment

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
                    tables.append((page_num, table.bbox, df))
    return tables

# Function to extract tables using Camelot
def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    tables = [(table.page, table._bbox, table.df) for table in tables]  # Convert to DataFrame
    return tables

# Function to extract tables using Tabula
def extract_tables_with_tabula(pdf_path):
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, pandas_options={'header': None})
    tables = [(i, None, table) for i, table in enumerate(tables, start=1)]
    return tables

# Function to extract text around tables using PyMuPDF
def extract_text_around_tables(pdf_path, margin=50):
    doc = fitz.open(pdf_path)
    texts_around_tables = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text_block, block_no, _ = block
            if text_block.strip():  # If the block has text
                texts_around_tables.append((page_num, text_block, (x0, y0, x1, y1)))
    return texts_around_tables

# Function to calculate the vertical distance between two bounding boxes
def calculate_distance(bbox1, bbox2):
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2
    return min(abs(y0_1 - y1_2), abs(y0_2 - y1_1))

# Function to associate tables with surrounding text and select the closest table
def associate_tables_with_text(tables, texts_around_tables, keywords):
    associated_tables = []
    for keyword in keywords:
        closest_table = None
        min_distance = float('inf')
        keyword_text = None
        keyword_bbox = None
        for page_num, text_block, bbox in texts_around_tables:
            if keyword.lower() in text_block.lower():
                for table_page, table_bbox, table in tables:
                    if table_page == page_num:
                        distance = calculate_distance(bbox, table_bbox)
                        if distance < min_distance:
                            min_distance = distance
                            closest_table = (text_block, table)
                            keyword_text = text_block
                            keyword_bbox = bbox
        if closest_table:
            associated_tables.append(closest_table)
    return associated_tables

# Function to evaluate the quality of extracted tables using heuristics
def evaluate_extraction(tables):
    scores = []
    for _, table in tables:
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
    for text, table in tables:
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
                
        cleaned_tables.append((text, table))
    return cleaned_tables

# Function to save extracted tables to Excel
def save_tables_to_excel(tables, output_path):
    if not tables:
        raise RuntimeError("No tables to save.")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, (associated_text, table) in enumerate(tables):
                sheet_name = f"Table_{i+1}"[:31]  # Excel sheet names must be 31 characters or less
                table.to_excel(writer, sheet_name=sheet_name, index=False)
                # Write the associated text as a comment in the first cell
                worksheet = writer.sheets[sheet_name]
                comment = Comment(associated_text, "Generated by script")
                worksheet.cell(1, 1).comment = comment
    except Exception as e:
        raise RuntimeError(f"Error saving tables to Excel: {e}")

# Main function to orchestrate the process
def main(pdf_path, output_excel_path, keywords):
    # Extract tables using different methods
    tables_pdfplumber = extract_tables_with_pdfplumber(pdf_path)
    tables_camelot = extract_tables_with_camelot(pdf_path)
    tables_tabula = extract_tables_with_tabula(pdf_path)

    # Extract surrounding text using PyMuPDF
    texts_around_tables = extract_text_around_tables(pdf_path)

    # Evaluate the quality of each extraction method using heuristics
    score_pdfplumber = evaluate_extraction(tables_pdfplumber)
    score_camelot = evaluate_extraction(tables_camelot)
    score_tabula = evaluate_extraction(tables_tabula)

    # Choose the best extraction method based on heuristic score
    best_method = max([(tables_pdfplumber, score_pdfplumber), 
                       (tables_camelot, score_camelot),
                       (tables_tabula, score_tabula)], key=lambda x: x[1])

    best_tables = best_method[0]

    # Associate tables with surrounding text and select the closest table
    associated_tables = associate_tables_with_text(best_tables, texts_around_tables, keywords)

    # Clean and format tables
    cleaned_tables = clean_and_format_tables(associated_tables)

    # Save cleaned tables to Excel
    save_tables_to_excel(cleaned_tables, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_path = 'path_to_your_pdf_file.pdf'
    output_excel_path = 'path_to_save_excel_file.xlsx'
    keywords = ['your', 'keywords', 'here']  # Add your specific keywords
    main(pdf_path, output_excel_path, keywords)
