import streamlit as st
from keybert import KeyBERT


@st.cache_resource
def load_keyword_model():
    return KeyBERT()


def extract_keywords(keyword_model, text, top_n=5):
    if not isinstance(text, str) or not text.strip():
        return ""  # Return empty string for non-string or empty inputs
    try:
        keywords = keyword_model.extract_keywords(text, top_n=top_n)
        return ", ".join([kw[0] for kw in keywords])
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""  # Return empty string if keyword extraction fails
