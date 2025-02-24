import re
import string
import gensim
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'https', '.com'])

def preprocess(text):
    """Cleans raw text data by removing URLs, punctuation, and stopwords."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text
