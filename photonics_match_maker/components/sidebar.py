"""
Sidebar component for the Photonics Matchmaker application.

This module defines the sidebar content and functionality.
"""

import streamlit as st
from components.faq import faq


def sidebar():
    """
    Display the sidebar content in the Streamlit app.
    """
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload a pdf, docx, or txt resume file :page_facing_up:\n"
            "2. Select the data source (Providers or Consumers)\n"
            "3. Adjust the number of matches to analyze\n"
            "4. Review the matches and detailed analysis\n"
        )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "The Photonics Matchmaker app uses advanced language models and "
            "embedding techniques to match professionals with companies in "
            "the photonics industry, providing data-driven recommendations."
        )
        faq()
