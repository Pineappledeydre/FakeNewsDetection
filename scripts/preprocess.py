import re
import string
import gensim
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'https', '.com'])

def wordopt(text):
    """Cleans raw text data by removing punctuation, URLs, and special characters."""
    text = text.lower()  
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\n', '', text)  
    return text

def preprocess(text):
    """Tokenizes and cleans text, removes stopwords, and returns a processed string."""
    cleaned_text = wordopt(text)
    result = [token for token in gensim.utils.simple_preprocess(cleaned_text)
              if token not in stop_words and len(token) > 3]
    return " ".join(result)
