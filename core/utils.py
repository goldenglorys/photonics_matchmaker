"""
Utility functions for the Photonics Matchmaker application.

This module contains helper functions for data preprocessing, text extraction,
embedding generation, and similarity calculation.
"""

from typing import Generator

import docx2txt
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_company_data_for_embedding(df):
    """
    Combine relevant company information into a single text field for embedding.

    Args:
        df (pd.DataFrame): The company dataframe.

    Returns:
        pd.DataFrame: The dataframe with an added 'combined_info' column.
    """
    df["combined_info"] = df.apply(
        lambda row: f"""
    Company Name: {row['Company Name']}
    Basic Company Information: {row['Basic Company Information']}
    Contact Information: {row['Contact Information']}
    Background: {row['Background']}
    Product and Service Portfolio: {row['Product and Service Portfolio']}
    Technology Focus and Expertise: {row['Technology Focus and Expertise']}
    Matching Criteria: {row['Matching Criteria']}
    Clientele and Partnerships: {row['Clientele and Partnerships']}
    Market Presence and Competitive Positioning: {row['Market Presence and Competitive Positioning']}
    Company Goals and Objectives: {row['Company Goals and Objectives']}
    """,
        axis=1,
    )
    return df


def clean_df(df):
    """
    Clean the dataframe by filling NaN values and ensuring text columns are strings.

    Args:
        df (pd.DataFrame): The company dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df = df.fillna("")
    text_columns = [
        "Technology Focus and Expertise",
        "Matching Criteria",
        "Product and Service Portfolio",
        "Company Goals and Objectives",
    ]
    for col in text_columns:
        df[col] = df[col].astype(str)
    return df


def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file.

    Args:
        pdf_file (file): The PDF file object.

    Returns:
        str: The extracted text content.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    """
    Extract text content from a DOCX file.

    Args:
        docx_file (file): The DOCX file object.

    Returns:
        str: The extracted text content.
    """
    text = docx2txt.process(docx_file)
    return text


def load_resume(uploaded_file):
    """
    Load and extract text from an uploaded resume file (PDF, DOCX, or TXT).

    Args:
        uploaded_file (UploadedFile): The uploaded file object.

    Returns:
        str: The extracted text content of the resume.
    """
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif (
        uploaded_file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return extract_text_from_docx(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8")


def generate_embeddings(model, texts):
    """
    Generate embeddings for a list of texts using the provided model.

    Args:
        model (SentenceTransformer): The embedding model.
        texts (list): A list of text strings to embed.

    Returns:
        np.array: The generated embeddings.
    """
    return model.encode(texts)


def calculate_similarities(candidate_embedding, company_embeddings):
    """
    Calculate cosine similarities between a candidate embedding and company embeddings.

    Args:
        candidate_embedding (np.array): The embedding of the candidate's resume.
        company_embeddings (np.array): The embeddings of company profiles.

    Returns:
        np.array: An array of similarity scores.
    """
    return cosine_similarity([candidate_embedding], company_embeddings)[0]


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """
    Generate a stream of chat responses from the Groq API.

    Args:
        chat_completion: The chat completion object from the Groq API.

    Yields:
        str: Chunks of the chat response content.
    """
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
