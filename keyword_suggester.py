import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from gensim.models import Word2Vec

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Data Collection
print("Loading the CSV file...")
df = pd.read_csv('financial_articles.csv')

# Display available columns
print("\nAvailable columns in your CSV file:")
for idx, column in enumerate(df.columns):
    print(f"{idx + 1}. {column}")

# Ask user to select the text column
while True:
    try:
        column_index = int(input("\nEnter the number of the column containing the text data: ")) - 1
        text_column = df.columns[column_index]
        break
    except (ValueError, IndexError):
        print("Invalid input. Please enter a valid number.")

# Data Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Apply preprocessing to each article
print(f"\nPreprocessing the '{text_column}' column...")
processed_articles = df[text_column].apply(preprocess_text)

# Vector Embedding with Word2Vec
print("\nTraining Word2Vec model...")
model = Word2Vec(sentences=processed_articles,
                 vector_size=100,
                 window=5,
                 min_count=1,
                 workers=4,
                 sg=1)

# Keyword Suggestion System
def suggest_keywords(input_word, top_n=5):
    try:
        similar_words = model.wv.most_similar(input_word, topn=top_n)
        return [word for word, _ in similar_words]
    except KeyError:
        return f"The word '{input_word}' is not in the vocabulary."

# Example usage
print("\nWelcome to the Keyword Suggestion System!")
print("Enter a word to get suggestions (or 'quit' to exit):")

while True:
    input_word = input("Enter a word: ").strip().lower()
    if input_word == 'quit':
        print("Thank you for using the Keyword Suggestion System. Goodbye!")
        break
    suggestions = suggest_keywords(input_word)
    print(f"Top 5 suggestions for '{input_word}': {suggestions}")
    print()