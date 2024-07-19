import os
import pandas as pd
import fitz  # PyMuPDF
import camelot
import pdfplumber
import tabula
import numpy as np
import logging
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.comments import Comment
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract tables using pdfplumber, including nested tables
def extract_tables_with_pdfplumber(pdf_path):
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc='Extracting tables with pdfplumber'), start=1):
                for table in page.extract_tables():
                    if table:  # Ensure table is not None
                        df = pd.DataFrame(table[1:], columns=table[0])
                        bbox = page.find_tables()[0].bbox if page.find_tables() else None
                        tables.append((page_num, bbox, df))
                        # Check for nested tables within each cell
                        for row in table:
                            for cell in row:
                                if isinstance(cell, str):
                                    nested_tables = extract_nested_tables_from_text(cell)
                                    if nested_tables:
                                        tables.extend([(page_num, None, nested_df) for nested_df in nested_tables])
    except Exception as e:
        logging.error(f"Error extracting tables with pdfplumber: {e}")
    return tables

# Function to detect and extract nested tables from text
def extract_nested_tables_from_text(text):
    nested_tables = []
    # Add heuristics or logic to detect and extract nested tables from text
    # For simplicity, let's assume nested tables are separated by new lines and commas
    lines = text.split('\n')
    for line in lines:
        if ',' in line:
            nested_data = [item.split(',') for item in lines if ',' in item]
            if nested_data:
                nested_df = pd.DataFrame(nested_data[1:], columns=nested_data[0])
                nested_tables.append(nested_df)
    return nested_tables

# Function to extract tables using Camelot
def extract_tables_with_camelot(pdf_path):
    tables = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        tables = [(int(table.page), table._bbox, table.df) for table in tables]  # Convert to DataFrame
    except Exception as e:
        logging.error(f"Error extracting tables with Camelot: {e}")
    return tables

# Function to extract tables using Tabula
def extract_tables_with_tabula(pdf_path):
    tables = []
    try:
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, pandas_options={'header': None})
        tables = [(i, None, table) for i, table in enumerate(tables, start=1)]
    except Exception as e:
        logging.error(f"Error extracting tables with Tabula: {e}")
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
        
        # Advanced cleaning: Detect and correct rotated text, handle merged cells, etc.
        table = detect_and_correct_rotated_text(table)
        
        # Convert data types if necessary (example: convert numeric columns)
        for col in table.columns:
            if isinstance(table[col], pd.Series):
                try:
                    table[col] = pd.to_numeric(table[col], errors='coerce')
                except ValueError:
                    logging.warning(f"Could not convert column {col} to numeric")
                
        cleaned_tables.append(table)  # Append just the table DataFrame
    return cleaned_tables

# Function to detect and correct rotated text (example implementation)
def detect_and_correct_rotated_text(table):
    # Add your logic to detect and correct rotated text in the table
    # This is just a placeholder example
    for col in table.columns:
        if table[col].dtype == 'object':
            table[col] = table[col].apply(lambda x: str(x).replace('rotated_text_example', 'corrected_text') if isinstance(x, str) else x)
    return table

# Function to merge tables that span across multiple pages
def merge_tables(tables):
    merged_tables = []
    prev_table = None
    for i, table in enumerate(tables):
        if prev_table is not None and (table.columns == prev_table.columns).all():
            prev_table = pd.concat([prev_table, table], ignore_index=True)
        else:
            if prev_table is not None:
                merged_tables.append(prev_table)
            prev_table = table
    if prev_table is not None:
        merged_tables.append(prev_table)
    return merged_tables

# Function to save extracted tables to Excel
def save_tables_to_excel(tables, output_path):
    if not tables:
        raise RuntimeError("No tables to save.")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, table in enumerate(tables):
                sheet_name = f"Table_{i+1}"[:31]  # Excel sheet names must be 31 characters or less
                table.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                
                # Autofit columns
                for col in worksheet.columns:
                    max_length = 0
                    column = col[0].column_letter  # Get the column name
                    for cell in col:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column].width = adjusted_width
    except Exception as e:
        logging.error(f"Error saving tables to Excel: {e}")
        raise RuntimeError(f"Error saving tables to Excel: {e}")

# Main function to orchestrate the process
def main(pdf_path, output_excel_path):
    logging.info("Starting table extraction with pdfplumber...")
    tables_pdfplumber = extract_tables_with_pdfplumber(pdf_path)
    if not tables_pdfplumber:
        logging.error("Pdfplumber extraction failed or returned no tables.")

    logging.info("Starting table extraction with Camelot...")
    tables_camelot = extract_tables_with_camelot(pdf_path)
    if not tables_camelot:
        logging.error("Camelot extraction failed or returned no tables.")

    logging.info("Starting table extraction with Tabula...")
    tables_tabula = extract_tables_with_tabula(pdf_path)
    if not tables_tabula:
        logging.error("Tabula extraction failed or returned no tables.")
    
    tables_list = [tables_pdfplumber, tables_camelot, tables_tabula]
    tables_list = [tables for tables in tables_list if tables]  # Filter out empty results

    if not tables_list:
        raise RuntimeError("All extraction methods failed or returned no tables.")
    
    logging.info("Evaluating extraction quality...")
    # Evaluate the quality of each extraction method using heuristics
    scores = [evaluate_extraction(tables) for tables in tables_list]
    
    logging.info("Choosing the best extraction method...")
    # Choose the best extraction method based on heuristic score
    best_index = np.argmax(scores)
    best_tables = tables_list[best_index]

    logging.info("Cleaning and formatting tables...")
    # Clean and format tables
    cleaned_tables = clean_and_format_tables(best_tables)

    logging.info("Merging tables...")
    # Merge tables that span across multiple pages
    merged_tables = merge_tables(cleaned_tables)

    logging.info("Saving tables to Excel...")
    # Save cleaned tables to Excel
    save_tables_to_excel(merged_tables, output_excel_path)
    logging.info(f"Tables successfully saved to {output_excel_path}")

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/file-xuDiR3z6kaLvXqzlsLozwU6k'
    output_excel_path = 'isitbetter.xlsx'
    main(pdf_path, output_excel_path)
