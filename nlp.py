import fitz  # PyMuPDF
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load QA model
qa_model_name = "distilbert-base-uncased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Load sentence transformer model for similarity search
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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

# Function to perform QA using transformers
def answer_question_with_qa_model(question, context):
    qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Function to find the most relevant paragraph for the question
def find_relevant_paragraph(question, paragraphs):
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)
    paragraph_embeddings = sentence_model.encode(paragraphs, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, paragraph_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return paragraphs[best_idx]

# Function to answer questions based on the analyzed data
def answer_question(question, text, analyzed_data):
    sentences = analyzed_data["Sentences"]
    
    # Split text into paragraphs for better context extraction
    paragraphs = text.split('\n\n')
    
    # Find the most relevant paragraph for the question
    relevant_paragraph = find_relevant_paragraph(question, paragraphs)
    
    # Answer the question using the QA model
    answer = answer_question_with_qa_model(question, relevant_paragraph)
    
    return answer

# Main function to orchestrate the process
def main(pdf_path, questions):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Analyze text using SpaCy
    analyzed_data = analyze_text(text)
    
    # Answer the questions
    answers = {}
    for question in questions:
        answer = answer_question(question, text, analyzed_data)
        answers[question] = answer
    
    return answers

# Example usage
if __name__ == "__main__":
    pdf_path = '/mnt/data/file-P4fJ4Rjb45dm72fLsbZshdRn'  # Use the uploaded file path
    questions = [
        "What are the key dates mentioned in the document?",
        "Which organizations are mentioned in the document?",
        "How many numbers are there in the document?",
        "Tell me more about the document content."
    ]
    
    answers = main(pdf_path, questions)
    for question, answer in answers.items():
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("\n" + "="*50 + "\n")
