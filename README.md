# Keyword Suggestion System

This project implements a simple keyword suggestion system using Word2Vec. It processes a dataset of text (e.g., financial articles) and suggests related words based on user input.

## Requirements

- Python 3.6+
- pandas
- nltk
- gensim

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/keyword-suggestion-system.git
   ```
2. Install the required packages:
   ```
   pip install pandas nltk gensim
   ```

## Usage

1. Place your CSV file (e.g., 'financial_articles.csv') in the project directory.
2. Run the script:
   ```
   python keyword_suggester.py
   ```
3. Follow the prompts to select the text column from your CSV.
4. Enter words to get suggestions. Type 'quit' to exit.

## How it Works

1. The script loads and preprocesses the text data from the CSV file.
2. It trains a Word2Vec model on the preprocessed text.
3. When you enter a word, it finds the most similar words according to the trained model.

## Note

The quality of suggestions depends on the content and size of your dataset. Larger, more diverse datasets in a specific domain will generally yield better results.
