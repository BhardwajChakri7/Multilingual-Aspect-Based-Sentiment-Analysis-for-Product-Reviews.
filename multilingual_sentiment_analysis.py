import spacy
from transformers import pipeline
import pandas as pd
import streamlit as st

class AspectSentimentModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def extract_aspects(self, text):
        doc = self.nlp(text)
        aspects = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]
        return aspects

    def analyze_sentiment(self, text):
        results = self.sentiment_analyzer(text)
        sentiment = results[0]['label']
        score = results[0]['score']
        return sentiment, score

    def analyze_review(self, review):
        aspects = self.extract_aspects(review)
        aspect_sentiments = {}
        for aspect in aspects:
            sentiment, score = self.analyze_sentiment(aspect)
            aspect_sentiments[aspect] = {'sentiment': sentiment, 'score': score}
        return aspect_sentiments

def process_file(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            model = AspectSentimentModel()
            df['aspect_sentiments'] = df['review'].apply(model.analyze_review)
            return df
        else:
            st.error("Uploaded CSV must contain a 'review' column.")
            return None
    else:
        st.error("Please upload a CSV file.")
        return None

def main():
    st.title("Multilingual Aspect-Based Sentiment Analysis for Product Reviews")
    uploaded_file = st.file_uploader("Upload your CSV file with reviews", type=["csv"])
    if uploaded_file is not None:
        df = process_file(uploaded_file)
        if df is not None:
            st.write(df)
            st.download_button(
                label="Download results as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='aspect_sentiments.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
