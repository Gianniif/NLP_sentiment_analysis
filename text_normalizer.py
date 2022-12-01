import re
import nltk
import spacy
import unicodedata
from unidecode import unidecode

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    text = BeautifulSoup(text,'html.parser').get_text()
    return text


def stem_text(text):
    porter = PorterStemmer()
    word = tokenizer.tokenize(text)
    text= " ".join(porter.stem(w)for w in word)
    return text


def lemmatize_text(text):
    text = nlp(text)
    text=" ".join([token.lemma_ for token in text])
    
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    for key , value in contraction_mapping.items():
        text = text.replace(key, value)
    return text


def remove_accented_chars(text):
    text = unidecode(text)
    return text


def remove_special_chars(text, remove_digits=False):
    if remove_digits == True:
       text = re.sub('[0-9]+', '', text)
    else :
        text = re.sub('[^a-zA-Z\s]', '', text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    separate_text = tokenizer.tokenize(text)
    
    tokens = [token.strip() for token in separate_text]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    text = ' '.join(filtered_tokens) 
    return text
            
    


def remove_extra_new_lines(text):
    
    text = text.replace('\n', " ")
    
    return text


def remove_extra_whitespace(text):
    
    text= " ".join(text.split())
    
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
