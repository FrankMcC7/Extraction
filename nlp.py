import fitz  # PyMuPDF
import spacy
import pandas as pd
import camelot
import tabula

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text and tables from PDF using PyMuPDF and table extraction libraries
def extract_text_and_tables_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    tables = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
        
        # Extract tables using camelot
        camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num+1), flavor='stream')
        for table in camelot_tables:
            if not table.df.empty:
                tables.append(table.df)
        
        # Extract tables using tabula
        tabula_tables = tabula.read_pdf(pdf_path, pages=page_num+1, multiple_tables=True)
        for table in tabula_tables:
            if not table.empty:
                tables.append(table)
    
    return text, tables

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

# Function to build tables from analyzed text
def build_tables_from_text(text_analysis):
    tables = []
    current_table = []
    
    for sentence in text_analysis["Sentences"]:
        if "Table" in sentence:  # Simple heuristic to detect table start
            if current_table:
                tables.append(pd.DataFrame(current_table))
                current_table = []
        current_table.append(sentence.split())  # Simple split, might need more sophisticated parsing
    
    if current_table:
        tables.append(pd.DataFrame(current_table))
    
    return tables

# Function to save extracted data to Excel
def save_data_to_excel(data, original_tables, rebuilt_tables, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Save sentences
        df_sentences = pd.DataFrame(data["Sentences"], columns=["Sentences"])
        df_sentences.to_excel(writer, sheet_name="Sentences", index=False)
        
        # Save entities
        df_entities = pd.DataFrame(data["Entities"], columns=["Entity", "Label"])
        df_entities.to_excel(writer, sheet_name="Entities", index=False)
        
        # Save numbers
        df_numbers = pd.DataFrame(data["Numbers"], columns=["Numbers"])
        df_numbers.to_excel(writer, sheet_name="Numbers", index=False)
        
        # Save dates
        df_dates = pd.DataFrame(data["Dates"], columns=["Dates"])
        df_dates.to_excel(writer, sheet_name="Dates", index=False)
        
        # Save organizations
        df_organizations = pd.DataFrame(data["Organizations"], columns=["Organizations"])
        df_organizations.to_excel(writer, sheet_name="Organizations", index=False)

        # Save original tables
        for i, table in enumerate(original_tables, start=1):
            table.to_excel(writer, sheet_name=f"Original_Table_{i}", index=False)

        # Save rebuilt tables
        for i, table in enumerate(rebuilt_tables, start=1):
            table.to_excel(writer, sheet_name=f"Rebuilt_Table_{i}", index=False)

# Main function to orchestrate the process
def main(pdf_path, output_excel_path):
    # Extract text and tables from PDF
    text, original_tables = extract_text_and_tables_from_pdf(pdf_path)
    
    # Analyze text using SpaCy
    analyzed_data = analyze_text(text)
    
    # Build tables from analyzed text
    rebuilt_tables = build_tables_from_text(analyzed_data)
    
    # Save analyzed data and tables to Excel
    save_data_to_excel(analyzed_data, original_tables, rebuilt_tables, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/file-P4fJ4Rjb45dm72fLsbZshdRn'  # Use the uploaded file path
    output_excel_path = '/mnt/data/analyzed_data_with_built_tables.xlsx'
    main(pdf_path, output_excel_path)
