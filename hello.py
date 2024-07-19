python -m spacy download en_core_web_sm
pip install spacy-*.whl
import spacy
nlp = spacy.load("en_core_web_sm")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
