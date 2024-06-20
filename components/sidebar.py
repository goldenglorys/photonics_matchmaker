import streamlit as st
from components.faq import faq


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload a pdf, docx, or txt file📄\n"
            "2. Ask a question about the document💬\n"
            "   Or you can ask the model to give you some questions about the document💬\n"
        )

        set_openai_api_key(st.secrets["GROQ_API_KEY"])

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖 The app allows you to ask questions about your "
            "documents and get accurate answers with instant citations. "
        )
        faq()
