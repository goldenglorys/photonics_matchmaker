# flake8: noqa
"""
FAQ component for the Photonics Matchmaker application.

This module defines the FAQ content to be displayed in the app.
"""


import streamlit as st


def faq():
    """
    Display the FAQ content in the Streamlit app.
    """
    st.markdown(
        """
# FAQ
## How does the embedding model mechanisms work?
By analyzing the semantic similarity between job seekers' resumes and company profiles, the system provides data-driven recommendations to streamline the job matching process.

Key Technical Points:

- Data Ingestion:
    Company data is sourced from Google Sheets, allowing for easy updates and maintenance.
    Professional resumes are uploaded directly to the application in various formats (PDF, DOCX, TXT).

- Text Preprocessing:
    Company information is consolidated into a comprehensive text representation.
    Resumes undergo text extraction to obtain their content regardless of the file format.

- Semantic Embedding:
    The system utilizes a pre-trained Sentence Transformer model (all-MiniLM-L6-v2) to convert text data into high-dimensional vector representations (embeddings).
    These embeddings capture the semantic meaning of the text, enabling nuanced comparisons.

- Similarity Computation:
    Cosine similarity is calculated between the resume embedding and each company's embedding.
    This mathematical measure quantifies the likeness of the professional's profile to each company's requirements and culture.

- Results Presentation:
    Companies are ranked based on their similarity scores, presented as percentages.
    Users can adjust the number of matches displayed, allowing for comprehensive or focused views.


## How does the language model mechanism work?
When you upload a document, it will be divided into smaller chunks 
and stored in a special type of database called a vector index 
that allows for semantic search and retrieval.

When you ask a question, it will search through the
document chunks and find the most relevant ones using the vector index.
Then, it will use ML model to generate a final answer.

## Is my data safe?
Yes, your data is safe. The app does not store your documents or
questions. All uploaded data is deleted after you close the browser tab.

## What do the numbers mean under each source?
For a PDF document, you will see a citation number like this: 3-12. 
The first number is the page number and the second number is 
the chunk number on that page. For DOCS and TXT documents, 
the first number is set to 1 and the second number is the chunk number.

## Are the answers 100% accurate?
No, the answers are not 100% accurate. It uses ML model to generate
answers. ML models are powerful language model, but it sometimes makes mistakes 
and is prone to hallucinations. Also, it uses semantic search
to find the most relevant chunks and does not see the entire document,
which means that it may not be able to find all the relevant information and
may not be able to answer all questions (especially summary-type questions
or questions that require a lot of context from the document).

But for most use cases, it is very accurate and can answer
most questions. Always check with the sources to make sure that the answers
are correct.
"""
    )
