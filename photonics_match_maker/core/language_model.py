"""
Language model module for the Photonics Matchmaker application.

This module handles interactions with the language model for detailed analysis.
"""


from typing import Generator

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq


def analyze_matches(top_matches, resume_text: str, model_name: str):
    """
    Analyze and display detailed results for top matches using a language model.

    Args:
        top_matches (pd.DataFrame): DataFrame containing top company matches.
        resume_text (str): The text content of the resume.
        model_name (str): The name of the language model to use.
    """
    llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model_name=model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant specializing in matching job candidates with companies in the photonics industry. Provide concise, insightful analyses focusing on the strengths of the match.",
            ),
            (
                "human",
                """
        Analyze the compatibility between the candidate and the following company in the photonics industry.
        
        Candidate Resume: {resume}
        
        Company Profile: {company_profile}
        
        Please provide the following:
        1. Compatibility Score: Give a score from 0-100 based on how well the candidate's skills and experience match the company's needs.
        2. Brief Explanation: In 2-3 sentences, explain why this company might be a good fit for the candidate.
        3. Key Strengths: List 2-3 bullet points highlighting the candidate's most relevant skills or experiences for this company.
        4. Potential Opportunities: Briefly mention 1-2 areas where the candidate could contribute to the company's goals or projects.

        Format your response as follows:
        Compatibility Score: [Score]/100
        Match Explanation: [Your brief explanation]
        Key Strengths:
        • [Strength 1]
        • [Strength 2]
        • [Strength 3]
        Potential Opportunities:
        • [Opportunity 1]
        • [Opportunity 2]
        """,
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    for _, company in top_matches.iterrows():
        st.write(
            f"### {company['Company Name']} (Match Score: {company['Match Score']:.2f}%)"
        )

        with st.spinner(
            f"Generating detailed analysis for {company['Company Name']}..."
        ):
            try:
                analysis = chain.invoke(
                    {
                        "resume": resume_text[:1000],
                        "company_profile": company["company_profile"],
                    }
                )
                st.write(analysis)
            except Exception as e:
                st.error(
                    f"Error in generating analysis for {company['Company Name']}: {str(e)}"
                )

        st.markdown("---")

    st.success("Analysis complete!")


def generate_chat_responses(
    client, prompt, model_option, max_tokens
) -> Generator[str, None, None]:
    """
    Generate chat responses using the Groq API.

    Args:
        prompt (str): The user's input prompt.
        model_option (str): The selected model option.
        max_tokens (int): The maximum number of tokens for the response.

    Yields:
        str: Chunks of the generated response.
    """
    chat_completion = client.chat.completions.create(
        model=model_option,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        + [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
