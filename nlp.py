import os
import pandas as pd
import fitz  # PyMuPDF
import spacy
from openpyxl import Workbook
from openpyxl.writer.excel import save_virtual_workbook

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to analyze and extract data using SpaCy
def analyze_text(text):
    doc = nlp(text)
    data = {
        "Sentences": [sent.text for sent in doc.sents],
        "Entities": [(ent.text, ent.label_) for ent in doc.ents],
        "Numbers": [token.text for token in doc if token.like_num],
        "Dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
        "Organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    }
    return data

# Function to save extracted data to Excel
def save_data_to_excel(data, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted Data"

    # Write sentences
    ws.append(["Sentences"])
    for sentence in data["Sentences"]:
        ws.append([sentence])
    ws.append([])  # Blank row for separation

    # Write entities
    ws.append(["Entities", "Label"])
    for entity, label in data["Entities"]:
        ws.append([entity, label])
    ws.append([])  # Blank row for separation

    # Write numbers
    ws.append(["Numbers"])
    for number in data["Numbers"]:
        ws.append([number])
    ws.append([])  # Blank row for separation

    # Write dates
    ws.append(["Dates"])
    for date in data["Dates"]:
        ws.append([date])
    ws.append([])  # Blank row for separation

    # Write organizations
    ws.append(["Organizations"])
    for org in data["Organizations"]:
        ws.append([org])

    # Save the workbook
    wb.save(output_path)

# Main function to orchestrate the process
def main(pdf_path, output_excel_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Analyze text using SpaCy
    data = analyze_text(text)
    
    # Save analyzed data to Excel
    save_data_to_excel(data, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/file-xxHEBAdBYpiFDAK8DlMi61ZQ'  # Use the uploaded file path
    output_excel_path = 'extracted_data.xlsx'
    main(pdf_path, output_excel_path)
