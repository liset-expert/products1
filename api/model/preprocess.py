import re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):

    """
    Preprocesses a given text by converting to lowercase, removing special characters,
    removing stopwords, tokenizing using ToktokTokenizer, and lemmatizing using WordNetLemmatizer.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text after applying lowercase conversion, special character removal,
             stopword removal, tokenization, and lemmatization.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Tokenization using ToktokTokenizer
    tokenizer = ToktokTokenizer()
    text = ' '.join(tokenizer.tokenize(' '.join(words)))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)
