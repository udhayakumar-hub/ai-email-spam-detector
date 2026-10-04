"""
NLP Text Preprocessing Module for AI Email Spam Detector.
Handles text cleaning, tokenization, stopword removal, and normalization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available or provide robust fallback
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Fallback common english stopwords if offline or download error
    STOPWORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
        'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
        'both', 'but', 'by', 'can', 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does',
        "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
        "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her',
        'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd",
        "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
        'let', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
        'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so',
        'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
        'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've",
        'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",
        'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when',
        "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",
        'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your',
        'yours', 'yourself', 'yourselves'
    }

stemmer = PorterStemmer()

# Regular expressions for cleaning
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
CURRENCY_PATTERN = re.compile(r'[\$£€¥₹]\s*\d+(?:,\d{3})*(?:\.\d+)?')
PUNCTUATION_PATTERN = re.compile(r'[' + re.escape(string.punctuation) + ']')
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """
    Perform raw text cleaning:
    - Lowercases text
    - Normalizes URLs, emails, phone numbers, and currency mentions
    - Strips punctuation and excess whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    
    # Replace URLs, emails, phone numbers, currency with token markers
    text = URL_PATTERN.sub(' urlmarker ', text)
    text = EMAIL_PATTERN.sub(' emailmarker ', text)
    text = PHONE_PATTERN.sub(' phonemarker ', text)
    text = CURRENCY_PATTERN.sub(' moneymarker ', text)
    
    # Remove punctuation
    text = PUNCTUATION_PATTERN.sub(' ', text)
    
    # Collapse multiple whitespaces
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text


def tokenize_and_remove_stopwords(text: str, apply_stemming: bool = True) -> str:
    """
    Tokenizes text, filters out stopwords and non-alphabetic tokens,
    and optionally applies Porter Stemming.
    """
    if not text:
        return ""
    
    words = text.split()
    processed_tokens = []
    
    for word in words:
        if word not in STOPWORDS and len(word) > 1:
            if apply_stemming:
                processed_tokens.append(stemmer.stem(word))
            else:
                processed_tokens.append(word)
                
    return " ".join(processed_tokens)


def preprocess_text(text: str, apply_stemming: bool = True) -> str:
    """
    Complete end-to-end preprocessing pipeline for a given text message.
    """
    cleaned = clean_text(text)
    tokenized = tokenize_and_remove_stopwords(cleaned, apply_stemming=apply_stemming)
    return tokenized
