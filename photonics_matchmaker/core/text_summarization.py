"""
Text summarization module for the Photonics Matchmaker application.

This module handles the summarization of text content.
"""

import streamlit as st
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer


@st.cache_resource
def load_summarizer():
    """
    Load and cache the LSA summarizer.

    Returns:
        LsaSummarizer: The loaded summarizer.
    """
    return LsaSummarizer()


def summarize_text(text: str, sentences_count: int = 2) -> str:
    """
    Summarize the given text.

    Args:
        text (str): The input text to summarize.
        sentences_count (int): The number of sentences in the summary.

    Returns:
        str: The summarized text.
    """
    summarizer = load_summarizer()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])
