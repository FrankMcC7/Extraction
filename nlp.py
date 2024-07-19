import os
import fitz  # PyMuPDF
import spacy
from collections import defaultdict

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

# Function to answer questions based on the analyzed data
def answer_question(question, analyzed_data):
    question_nlp = nlp(question)
    entities = analyzed_data["Entities"]
    sentences = analyzed_data["Sentences"]

    answer = []
    if any(token.lemma_ == "number" or token.lemma_ == "amount" for token in question_nlp):
        answer = analyzed_data["Numbers"]
    elif any(token.lemma_ == "date" or token.lemma_ == "time" for token in question_nlp):
        answer = analyzed_data["Dates"]
    elif any(token.lemma_ == "organization" or token.lemma_ == "company" for token in question_nlp):
        answer = analyzed_data["Organizations"]
    else:
        # Find relevant sentences
        question_tokens = set(token.lemma_ for token in question_nlp)
        for sent in sentences:
            sent_nlp = nlp(sent)
            sent_tokens = set(token.lemma_ for token in sent_nlp)
            if question_tokens & sent_tokens:
                answer.append(sent)
    
    return answer if answer else ["Sorry, I couldn't find an answer to your question."]

# Main function to orchestrate the process
def main(pdf_path, question):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Analyze text using SpaCy
    analyzed_data = analyze_text(text)
    
    # Answer the question
    answer = answer_question(question, analyzed_data)
    return answer

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/file-P4fJ4Rjb45dm72fLsbZshdRn'  # Use the uploaded file path
    question = "What are the key dates mentioned in the document?"
    answer = main(pdf_path, question)
    for ans in answer:
        print(ans)
