import fitz  # PyMuPDF
import spacy
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Function to calculate cosine similarity between the question and sentences
def find_most_relevant_sentences(question, sentences):
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]  # Ignore self-similarity
    most_relevant_indices = similarity_scores.argsort()[-5:][::-1]  # Top 5 sentences
    return [sentences[i] for i in most_relevant_indices]

# Function to answer questions based on the analyzed data
def answer_question(question, analyzed_data):
    question_nlp = nlp(question)
    entities = analyzed_data["Entities"]
    sentences = analyzed_data["Sentences"]

    answer = []
    if any(token.lemma_ in ["number", "amount"] for token in question_nlp):
        answer = analyzed_data["Numbers"]
    elif any(token.lemma_ in ["date", "time"] for token in question_nlp):
        answer = analyzed_data["Dates"]
    elif any(token.lemma_ in ["organization", "company"] for token in question_nlp):
        answer = analyzed_data["Organizations"]
    else:
        # Find relevant sentences using cosine similarity
        relevant_sentences = find_most_relevant_sentences(question, sentences)
        answer = relevant_sentences
    
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
    questions = [
        "What are the key dates mentioned in the document?",
        "Which organizations are mentioned in the document?",
        "How many numbers are there in the document?",
        "Tell me more about the document content."
    ]
    
    for question in questions:
        print(f"Question: {question}")
        answers = main(pdf_path, question)
        for ans in answers:
            print(ans)
        print("\n" + "="*50 + "\n")
