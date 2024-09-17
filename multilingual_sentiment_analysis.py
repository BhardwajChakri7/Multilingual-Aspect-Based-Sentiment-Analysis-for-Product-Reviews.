import spacy
from transformers import pipeline
import pandas as pd

# Load spaCy models for multilingual aspect extraction
nlp_en = spacy.load("en_core_web_sm")
nlp_hi = spacy.blank("hi")
nlp_ta = spacy.blank("ta")
nlp_te = spacy.blank("te")

# Sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Function to extract aspects and sentiments
def analyze_review(review, language):
    if language == 'en':
        doc = nlp_en(review)
    elif language == 'hi':
        doc = nlp_hi(review)
    elif language == 'ta':
        doc = nlp_ta(review)
    elif language == 'te':
        doc = nlp_te(review)
    else:
        return []

    aspects = []
    sentiments = []
    for chunk in doc.noun_chunks:
        aspect = chunk.text
        result = sentiment_analysis(aspect)
        sentiment = result[0]['label']
        aspects.append(aspect)
        sentiments.append(sentiment)
    
    return pd.DataFrame({'Aspect': aspects, 'Sentiment': sentiments})

# Example usage
if __name__ == "__main__":
    review = "This phone has a great camera but the battery life is short."
    language = 'en'
    result = analyze_review(review, language)
    print(result)
