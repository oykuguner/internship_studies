import pandas as pd
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
spell = SpellChecker()

def preprocess_text(text):
    # küçük harfe dönüştürme
    text = text.lower()
    
    # noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    
    # durak kelimeleri kaldırma
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # yazım denetimi
    corrected_text = []
    for word in text.split():
        corrected_word = spell.correction(word)
        # Eğer düzeltme None döndürürse orijinal kelimeyi kullan
        corrected_text.append(corrected_word if corrected_word is not None else word)
    
    return text

def main():
    dataset = load_dataset("EgehanEralp/imdb-single-sentence", split="train")

    df = pd.DataFrame(dataset)

    df['text'] = df['text'].apply(preprocess_text)

    print(df.head())

if __name__ == "__main__":
    main()
