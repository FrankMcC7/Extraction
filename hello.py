python -m spacy download en_core_web_sm
pip install spacy-*.whl
import spacy
nlp = spacy.load("en_core_web_sm")
