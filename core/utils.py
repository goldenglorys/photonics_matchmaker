from typing import Generator

import docx2txt
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_company_data_for_embedding(df):
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
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text


def load_resume(uploaded_file):
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
    return model.encode(texts)


def calculate_similarities(candidate_embedding, company_embeddings):
    return cosine_similarity([candidate_embedding], company_embeddings)[0]


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
