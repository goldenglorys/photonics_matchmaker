import nltk
import streamlit as st
from components.sidebar import sidebar
from core.keyword import extract_keywords, load_keyword_model
from core.summarizer import load_summarizer, summarize_text
from core.utils import (calculate_similarities, clean_df,
                        generate_chat_responses, generate_embeddings,
                        load_resume)
from groq import Groq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from streamlit_gsheets import GSheetsConnection

st.set_page_config(
    page_icon="🚀", layout="wide", page_title="Photonics Matchmaker Goes Brrrrrrrr..."
)

sidebar()

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

# Load companies data based on category
provider_df = conn.read(worksheet="TECH_PROVIDERS", ttl="30m")
consumer_df = conn.read(worksheet="TECH_CONSUMERS", ttl="30m")

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Download the punkt resource if it's not already present
if nltk.data.find("tokenizers/punkt") is None:
    nltk.download("punkt")

embedding_model = load_embedding_model()


summarizer = load_summarizer()


keyword_model = load_keyword_model()


# Preprocess company data
@st.cache_data
def create_condensed_profile(row):
    tech_focus = summarize_text(summarizer, str(row["Technology Focus and Expertise"]))
    matching_criteria = extract_keywords(keyword_model, str(row["Matching Criteria"]))
    products_services = summarize_text(
        summarizer, str(row["Product and Service Portfolio"])
    )
    company_goals = extract_keywords(
        keyword_model, str(row["Company Goals and Objectives"])
    )

    profile = f"""
    Company: {row['Company Name']}
    Focus: {tech_focus}
    Key Criteria: {matching_criteria}
    Products/Services: {products_services}
    Goals: {company_goals}
    """
    return profile.strip()


@st.cache_data
def preprocess_company_data(df):
    df["company_profile"] = df.apply(create_condensed_profile, axis=1)
    return df


keyword_provider_df = preprocess_company_data(clean_df(provider_df))
keyword_consumer_df = preprocess_company_data(clean_df(consumer_df))


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🚀")

st.subheader(
    "Photonics Matchmaker | Match professionals with companies in the photonics industry.",
    divider="rainbow",
    anchor=False,
)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)


def load_and_process_data(data_source):
    if data_source == "Providers":
        return keyword_provider_df
    else:
        return keyword_consumer_df


def process_resume(uploaded_file):
    if uploaded_file is not None:
        resume_text = load_resume(uploaded_file)
        st.write(resume_text[:500] + "...")
        return resume_text
    return None


def calculate_matches(data, resume_text, embedding_model):
    company_embeddings = generate_embeddings(
        embedding_model, data["company_profile"].tolist()
    )
    resume_embedding = generate_embeddings(embedding_model, [resume_text])[0]
    similarities = calculate_similarities(resume_embedding, company_embeddings)
    data["Match Score"] = similarities * 100
    return data.sort_values("Match Score", ascending=False)


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
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
}


chatbot_tab, ml_tab, emb_vec_repr_tab, provider_df_tab, consumer_df_tab = st.tabs(
    [
        "General ChatBot",
        "Use Language Model",
        "Use Embedding Model + Vector",
        "Providers Data",
        "Consumers Data",
    ]
)

with chatbot_tab:
    st.markdown(
        """
Use varieties of open source models
"""
    )
    st.markdown("""---""")

    # Layout for model selection and max_tokens slider
    col1, col2 = st.columns(2)

    # Create a container for chat history
    chat_container = st.container()

    with col1:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=0,  # Default to the first model in the list
        )

    # Detect model change and clear chat history if model has changed
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    with col2:
        # Adjust max_tokens slider dynamically based on the selected model
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,  # Minimum value to allow some flexibility
            max_value=max_tokens_range,
            # Default value or max allowed if less
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}",
        )

    prompt = st.chat_input("Enter your prompt here...")

    with chat_container:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user", avatar="👨‍💻"):
                st.markdown(prompt)

            # Fetch response from Groq API
            try:
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                )

                # Use the generator function with st.write_stream
                with st.chat_message("assistant", avatar="🤖"):
                    chat_responses_generator = generate_chat_responses(chat_completion)
                    full_response = st.write_stream(chat_responses_generator)
            except Exception as e:
                st.error(e, icon="🚨")

            # Append the full response to session_state.messages
            if isinstance(full_response, str):
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                # Handle the case where full_response is not a string
                combined_response = "\n".join(str(item) for item in full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": combined_response}
                )


with ml_tab:
    st.markdown(
        """
    This tab uses a combination of embedding similarity and language model analysis for matchmaking.
    """
    )
    st.markdown("""---""")

    # Layout for data source, number of matches, and model selection
    col1, col2 = st.columns(2)

    with col1:
        data_source = st.selectbox(
            "Choose a data source:", options=["Providers", "Consumers"], key="data_source"
        )
        data = load_and_process_data(data_source)

    with col2:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            key="language_model_option",
            format_func=lambda x: models[x]["name"],
            index=0,  # Default to the first model in the list
        )

    uploaded_file = st.file_uploader(
        "Choose a resume file", type=["pdf", "docx", "txt"], key="language_model_uploader"
    )
    resume_text = process_resume(uploaded_file)

    if resume_text is not None and not data.empty:
        sorted_companies = calculate_matches(data, resume_text, embedding_model)

        col3, col4 = st.columns(2)

        with col3:
            num_matches = st.slider(
                "Number of companies to analyze", min_value=1, max_value=10, value=5
            )

        with col4:
            max_tokens = st.slider(
                "Max Tokens:",
                min_value=512,
                max_value=models[model_option]["tokens"],
                value=min(32768, models[model_option]["tokens"]),
                step=512,
                help=f"Adjust the maximum number of tokens for the model's response. Max for selected model: {models[model_option]['tokens']}",
            )

        top_matches = sorted_companies.head(num_matches)

        llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model_name=model_option)

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

with emb_vec_repr_tab:
    st.markdown(
        """
        This mechanism uses embedding models to convert company and candidate information into vectors, then use cosine similarity to find the best matches. This can be more efficient than using a large language model for every comparison.
    """
    )
    st.markdown("""---""")
    data_source = st.selectbox(
        "Choose a data source:", options=["Providers", "Consumers"]
    )
    data = load_and_process_data(data_source)

    uploaded_file = st.file_uploader(
        "Choose a resume file", type=["pdf", "docx", "txt"], key="embed_model_uploader"
    )
    resume_text = process_resume(uploaded_file)

    if resume_text is not None and not data.empty:
        sorted_companies = calculate_matches(data, resume_text, embedding_model)

        st.subheader("Matches:")
        num_matches = st.slider(
            "Number of matches to display",
            min_value=1,
            max_value=len(sorted_companies),
            value=10,
        )
        display_matches(sorted_companies, num_matches)

with provider_df_tab:
    st.dataframe(keyword_provider_df, height=1000, use_container_width=True)

with consumer_df_tab:
    st.dataframe(keyword_consumer_df, height=1000, use_container_width=True)
