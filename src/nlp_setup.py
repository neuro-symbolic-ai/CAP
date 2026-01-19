import benepar
import spacy

# Initialize spaCy model with Benepar
nlp = spacy.load("en_core_web_sm")
if not spacy.util.is_package("benepar"):
    benepar.download('benepar_en3')
nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)