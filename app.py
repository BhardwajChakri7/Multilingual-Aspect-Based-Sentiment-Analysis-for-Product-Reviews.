import streamlit as st
import pandas as pd
from multilingual_sentiment_analysis import analyze_review

# Page config
st.set_page_config(
    page_title="Multilingual Aspect-Based Sentiment Analysis",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title
st.title("Multilingual Aspect-Based Sentiment Analysis for Product Reviews")

# Language selection
language = st.selectbox("Select the language of the review:", ["en", "hi", "ta", "te"])

# Text input
review = st.text_area("Enter the product review:")

# Analyze button
if st.button("Analyze"):
    if review:
        result = analyze_review(review, language)
        if not result.empty:
            st.write("### Analysis Results")
            st.dataframe(result)
        else:
            st.write("No aspects were found in the review.")
    else:
        st.write("Please enter a review to analyze.")
