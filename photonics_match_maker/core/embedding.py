"""
Embedding module for the Photonics Matchmaker application.

This module handles the generation of embeddings and calculation of similarities.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache the sentence transformer model for generating embeddings.

    Args:
        model_name (str): The name of the embedding model to load.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(model: SentenceTransformer, texts: list[str]):
    """
    Generate embeddings for a list of texts using the provided model.

    Args:
        model (SentenceTransformer): The embedding model.
        texts (list[str]): A list of text strings to embed.

    Returns:
        np.ndarray: The generated embeddings.
    """
    return model.encode(texts)


def calculate_similarities(candidate_embedding, company_embeddings):
    """
    Calculate cosine similarities between a candidate embedding and company embeddings.

    Args:
        candidate_embedding (np.ndarray): The embedding of the candidate's resume.
        company_embeddings (np.ndarray): The embeddings of company profiles.

    Returns:
        np.ndarray: An array of similarity scores.
    """
    return cosine_similarity([candidate_embedding], company_embeddings)[0]


def calculate_matches(data, resume_text: str):
    """
    Calculate similarity scores between a resume and company profiles.

    Args:
        data (pd.DataFrame): The company dataframe.
        resume_text (str): The text content of the resume.

    Returns:
        pd.DataFrame: The company dataframe sorted by match score in descending order.
    """
    embedding_model = load_embedding_model()
    company_embeddings = generate_embeddings(
        embedding_model, data["company_profile"].tolist()
    )
    resume_embedding = generate_embeddings(embedding_model, [resume_text])[0]
    similarities = calculate_similarities(resume_embedding, company_embeddings)
    data["Match Score"] = similarities * 100
    return data.sort_values("Match Score", ascending=False)
