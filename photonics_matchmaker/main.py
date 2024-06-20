"""
Photonics Matchmaker: A Streamlit application for matching professionals with companies in the photonics industry.

This application uses embedding models and language models to analyze resumes and company profiles,
providing matchmaking services and detailed analyses of potential matches.
"""

import logging

import nltk
import streamlit as st
from groq import Groq

from photonics_matchmaker.components.sidebar import sidebar
from photonics_matchmaker.core.data_processing import load_and_process_data
from photonics_matchmaker.core.embedding import calculate_matches
from photonics_matchmaker.core.language_model import (analyze_matches,
                                                      generate_chat_responses)
from photonics_matchmaker.utils.helpers import load_icon, process_resume

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Download the punkt resource if it's not already present
if nltk.data.find("tokenizers/punkt") is None:
    nltk.download("punkt")


def main():
    try:
        st.set_page_config(
            page_icon="🚀",
            layout="wide",
            page_title="Photonics Matchmaker Goes Brrrrrrrr...",
        )

        sidebar()

        load_icon("🚀")

        st.title("Photonics Matchmaker")
        st.subheader(
            "Match professionals with companies in the photonics industry",
            divider="rainbow",
        )

        # Initialize chat history and selected model
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None

        # Define model details
        models = {
            "llama3-70b-8192": {
                "name": "LLaMA3-70b-Instruct",
                "tokens": 8192,
                "developer": "Meta",
            },
            "llama3-8b-8192": {
                "name": "LLaMA3-8b-Instruct",
                "tokens": 8192,
                "developer": "Meta",
            },
            "mixtral-8x7b-32768": {
                "name": "Mixtral-8x7b-Instruct-v0.1",
                "tokens": 32768,
                "developer": "Mistral",
            },
            "gemma-7b-it": {
                "name": "Gemma-7b-it",
                "tokens": 8192,
                "developer": "Google",
            },
        }

        # Create tabs
        (
            chatbot_tab,
            ml_tab,
            emb_vec_repr_tab,
            provider_df_tab,
            consumer_df_tab,
        ) = st.tabs(
            [
                "General ChatBot",
                "Use Language Model",
                "Use Embedding Model + Vector",
                "Providers Data",
                "Consumers Data",
            ]
        )

        with chatbot_tab:
            st.markdown("Use varieties of open source models")
            st.markdown("---")

            col1, col2 = st.columns(2)

            chat_container = st.container()

            with col1:
                model_option = st.selectbox(
                    "Choose a model:",
                    options=list(models.keys()),
                    format_func=lambda x: models[x]["name"],
                    index=0,
                )

            if st.session_state.selected_model != model_option:
                st.session_state.messages = []
                st.session_state.selected_model = model_option

            max_tokens_range = models[model_option]["tokens"]

            with col2:
                max_tokens = st.slider(
                    "Max Tokens:",
                    min_value=512,
                    max_value=max_tokens_range,
                    value=min(32768, max_tokens_range),
                    step=512,
                    help=f"Adjust the maximum number of tokens for the model's response. Max for selected model: {max_tokens_range}",
                )

            prompt = st.chat_input("Enter your prompt here...")

            with chat_container:
                for message in st.session_state.messages:
                    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])

                if prompt:
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    with st.chat_message("user", avatar="👨‍💻"):
                        st.markdown(prompt)

                    try:
                        with st.chat_message("assistant", avatar="🤖"):
                            full_response = st.write_stream(
                                generate_chat_responses(
                                    client, prompt, model_option, max_tokens
                                )
                            )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_response}
                        )
                    except Exception as e:
                        st.error(e, icon="🚨")

        with ml_tab:
            st.markdown(
                "This tab uses a combination of embedding similarity and language model analysis for matchmaking."
            )
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                data_source = st.selectbox(
                    "Choose a data source:",
                    options=["Providers", "Consumers"],
                    key="ml_data_source",
                )
                data = load_and_process_data(data_source)

            with col2:
                model_option = st.selectbox(
                    "Choose a model:",
                    options=list(models.keys()),
                    key="language_model_option",
                    format_func=lambda x: models[x]["name"],
                    index=0,
                )

            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=["pdf", "docx", "txt"],
                key="language_model_uploader",
            )
            resume_text = process_resume(uploaded_file)

            if resume_text and not data.empty:
                sorted_companies = calculate_matches(data, resume_text)

                col3, col4 = st.columns(2)

                with col3:
                    num_matches = st.slider(
                        "Number of companies to analyze",
                        key="num_matches",
                        min_value=1,
                        max_value=10,
                        value=5,
                    )

                with col4:
                    max_tokens = st.slider(
                        "Max Tokens:",
                        key="max_tokens",
                        min_value=512,
                        max_value=models[model_option]["tokens"],
                        value=min(32768, models[model_option]["tokens"]),
                        step=512,
                        help=f"Adjust the maximum number of tokens for the model's response. Max for selected model: {models[model_option]['tokens']}",
                    )

                top_matches = sorted_companies.head(num_matches)
                analyze_matches(top_matches, resume_text, model_option)

        with emb_vec_repr_tab:
            st.markdown(
                "This mechanism uses embedding models to convert company and candidate information into vectors, then use cosine similarity to find the best matches."
            )
            st.markdown("---")

            data_source = st.selectbox(
                "Choose a data source:",
                options=["Providers", "Consumers"],
                key="emb_data_source",
            )
            data = load_and_process_data(data_source)

            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=["pdf", "docx", "txt"],
                key="embed_model_uploader",
            )
            resume_text = process_resume(uploaded_file)

            if resume_text and not data.empty:
                sorted_companies = calculate_matches(data, resume_text)

                st.subheader("Matches:")
                num_matches = st.slider(
                    "Number of matches to display",
                    min_value=1,
                    max_value=len(sorted_companies),
                    value=10,
                )
                display_matches(sorted_companies, num_matches)

        with provider_df_tab:
            provider_data = load_and_process_data("Providers")
            st.dataframe(provider_data, height=1000, use_container_width=True)

        with consumer_df_tab:
            consumer_data = load_and_process_data("Consumers")
            st.dataframe(consumer_data, height=1000, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)


def display_matches(sorted_companies, num_matches):
    display_df = sorted_companies[
        [
            "Company Name",
            "Match Score",
            "Contact Information",
            "Basic Company Information",
        ]
    ].copy()
    display_df["Match Score"] = display_df["Match Score"].apply(lambda x: f"{x:.2f}%")
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1
    st.table(display_df.head(num_matches))


if __name__ == "__main__":
    main()
