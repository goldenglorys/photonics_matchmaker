"""
Helper functions for the Photonics Matchmaker application.

This module contains utility functions for file processing and other helper tasks.
"""

from typing import BinaryIO

import docx2txt
import PyPDF2
import streamlit as st


def load_resume(uploaded_file: BinaryIO) -> str:
    """
    Load and extract text from an uploaded resume file (PDF, DOCX, or TXT).

    Args:
        uploaded_file (BinaryIO): The uploaded file object.

    Returns:
        str: The extracted text content of the resume.

    Raises:
        ValueError: If the file type is not supported.
    """
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_file (BinaryIO): The PDF file object.

    Returns:
        str: The extracted text content.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file: BinaryIO) -> str:
    """
    Extract text content from a DOCX file.

    Args:
        docx_file (BinaryIO): The DOCX file object.

    Returns:
        str: The extracted text content.
    """
    text = docx2txt.process(docx_file)
    return text


def process_resume(uploaded_file) -> str:
    """
    Process an uploaded resume file and display a preview.

    Args:
        uploaded_file (UploadedFile): The uploaded resume file.

    Returns:
        str: The extracted text from the resume, or None if no file was uploaded.
    """
    if uploaded_file is not None:
        resume_text = load_resume(uploaded_file)
        st.write("Resume Preview:")
        st.write(resume_text[:500] + "...")
        return resume_text
    return None


def load_icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )
