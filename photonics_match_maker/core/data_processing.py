"""
Data processing module for the Photonics Matchmaker application.

This module handles loading and preprocessing of company data.
"""

import streamlit as st
from core.keyword_extraction import extract_keywords
from core.text_summarization import summarize_text
from streamlit_gsheets import GSheetsConnection


@st.cache_data
def load_and_process_data(data_source: str):
    """
    Load and process company data based on the selected data source.

    Args:
        data_source (str): Either "Providers" or "Consumers".

    Returns:
        pd.DataFrame: The processed company dataframe.
    """
    conn = st.connection("gsheets", type=GSheetsConnection)

    if data_source == "Providers":
        df = conn.read(worksheet="TECH_PROVIDERS", ttl="30m")
    else:
        df = conn.read(worksheet="TECH_CONSUMERS", ttl="30m")

    df = clean_df(df)
    df = preprocess_company_data(df)
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


@st.cache_data
def preprocess_company_data(df):
    """
    Preprocess the company dataframe by creating condensed profiles for each company.

    Args:
        df (pd.DataFrame): The company dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe with added 'company_profile' column.
    """
    df["company_profile"] = df.apply(create_condensed_profile, axis=1)
    return df


def create_condensed_profile(row) -> str:
    """
    Create a condensed company profile from a dataframe row.

    Args:
        row (pd.Series): A row from the company dataframe.

    Returns:
        str: A condensed profile string containing key company information.
    """
    tech_focus = summarize_text(str(row["Technology Focus and Expertise"]))
    matching_criteria = extract_keywords(str(row["Matching Criteria"]))
    products_services = summarize_text(str(row["Product and Service Portfolio"]))
    company_goals = extract_keywords(str(row["Company Goals and Objectives"]))

    profile = f"""
    Company: {row['Company Name']}
    Focus: {tech_focus}
    Key Criteria: {matching_criteria}
    Products/Services: {products_services}
    Goals: {company_goals}
    """
    return profile.strip()
