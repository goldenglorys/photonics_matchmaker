"""
Keyword extraction module for the Photonics Matchmaker application.

This module handles the extraction of keywords from text.
"""

import streamlit as st
from keybert import KeyBERT


@st.cache_resource
def load_keyword_model():
    """
    Load and cache the KeyBERT model for keyword extraction.

    Returns:
        KeyBERT: The loaded KeyBERT model.
    """
    return KeyBERT()


def extract_keywords(text: str, top_n: int = 5) -> str:
    """
    Extract keywords from the given text.

    Args:
        text (str): The input text to extract keywords from.
        top_n (int): The number of top keywords to extract.

    Returns:
        str: A comma-separated string of extracted keywords.
    """
    keyword_model = load_keyword_model()
    if not isinstance(text, str) or not text.strip():
        return ""  # Return empty string for non-string or empty inputs
    try:
        keywords = keyword_model.extract_keywords(text, top_n=top_n)
        return ", ".join([kw[0] for kw in keywords])
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""  # Return empty string if keyword extraction fails
