import streamlit as st
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer


# Text summarization functions
@st.cache_resource
def load_summarizer():
    return LsaSummarizer()


def summarize_text(summarizer, text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])
