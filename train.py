import fitz  # PyMuPDF
import spacy
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_trf")  # Using the transformer model for better embeddings

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    
    return text

# Function to preprocess text using SpaCy
def preprocess_text(text):
    doc = nlp(text)
    
    # Extract tokens, lemmas, entities, and noun chunks
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    entities = [ent.text for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    combined_text = tokens + entities + noun_chunks
    return " ".join(combined_text)

# Function to process multiple PDFs and collect text data
def process_bulk_pdfs(pdf_dir):
    texts = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            preprocessed_text = preprocess_text(text)
            texts.append(preprocessed_text)
    
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
    
    return kmeans, vectorizer, X

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

# Function to determine the optimal number of clusters using the Elbow method
def plot_elbow_method(X):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

# Function to determine the optimal number of clusters using the Silhouette score
def plot_silhouette_scores(X):
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Main function to orchestrate the process
def main(pdf_dir, output_excel_path, num_clusters=5):
    # Process multiple PDFs and collect text data
    texts = process_bulk_pdfs(pdf_dir)
    
    # Vectorize text data and apply clustering
    kmeans, vectorizer, X = cluster_texts(texts, num_clusters)
    
    # Plot Elbow method and Silhouette scores
    plot_elbow_method(X)
    plot_silhouette_scores(X)
    
    # Save clustered text data to Excel
    save_clustered_data_to_excel(texts, kmeans, vectorizer, output_excel_path)

# Example usage
if __name__ == "__main__":
    pdf_dir = '/mnt/data'  # Set this to your directory containing PDF files
    output_excel_path = '/mnt/data/analyzed_data_clusters.xlsx'
    main(pdf_dir, output_excel_path, num_clusters=5)
