import fitz  # PyMuPDF
import spacy
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Function to process multiple PDFs and collect text data
def process_bulk_pdfs(pdf_dir):
    texts = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            texts.append(text)
    
    return texts

# Function to vectorize text data and apply clustering
def cluster_texts(texts, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2, random_state=0)
    reduced_data = pca.fit_transform(X.toarray())
    
    # Plot the clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_)
    plt.title('Cluster visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    
    return kmeans, vectorizer

# Function to save clustered text data to Excel
def save_clustered_data_to_excel(texts, kmeans, vectorizer, output_path):
    cluster_labels = kmeans.labels_
    cluster_data = {i: [] for i in range(kmeans.n_clusters)}
    
    for text, label in zip(texts, cluster_labels):
        cluster_data[label].append(text)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for cluster_id, texts in cluster_data.items():
            df = pd.DataFrame(texts, columns=["Text"])
            df.to_excel(writer, sheet_name=f"Cluster_{cluster_id}", index=False)

# Main function to orchestrate the process
def main(pdf_dir, output_excel_path, num_clusters=5):
    # Process multiple PDFs and collect text data
    texts = process_bulk_pdfs(pdf_dir)
    
    # Vectorize text data and apply clustering
    kmeans, vectorizer = cluster_texts(texts, num_clusters)
    
    # Save clustered text data to Excel
    save_clustered_data_to_excel(texts, kmeans, vectorizer, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_dir = '/path/to/your/pdf/folder'  # Set this to your directory containing PDF files
    output_excel_path = '/path/to/output/analyzed_data_clusters.xlsx'
    main(pdf_dir, output_excel_path, num_clusters=5)
