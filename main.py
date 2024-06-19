import streamlit as st
import streamlit.components.v1 as components
from components.sidebar import sidebar
from groq import Groq
from sentence_transformers import SentenceTransformer
from streamlit_gsheets import GSheetsConnection
from utils import (calculate_similarities, generate_chat_responses,
                   generate_embeddings, load_resume, preprocess_company_data)

st.set_page_config(
    page_icon="🚀", layout="wide", page_title="Photonics Matchmaker Goes Brrrrrrrr..."
)

# Auto-scrolling script
auto_scroll_script = """
<script>
    var chatContainer = window.parent.document.querySelector('[data-testid="stExpander"] .main');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
</script>
"""

components.html(auto_scroll_script, height=0)

sidebar()

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

provider_df = conn.read(worksheet="TECH_PROVIDERS", ttl="30m")
consumer_df = conn.read(worksheet="TECH_CONSUMERS", ttl="30m")

# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


model = load_model()

icon("🚀")

st.subheader(
    "Photonics Matchmaker | Match professionals with companies in the photonics industry.",
    divider="rainbow",
    anchor=False,
)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
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
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
}


chatbot_tab, ml_tab, emb_vec_repr_tab, provider_df_tab, consumer_df_tab  = st.tabs(
    [
        "General ChatBot",
        "Use Language Model",
        "Use Embedding Model + Vector",
        "Providers Data",
        "Consumers Data"
    ]
)

with chatbot_tab:
    st.markdown(
        """
Use varieties of open source models from Groq
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

        if prompt :
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
    pass

with emb_vec_repr_tab:
    st.markdown(
        """
        This mechanism uses embedding models to convert company and candidate information into vectors, then use cosine similarity to find the best matches. This can be more efficient than using a large language model for every comparison.
    """
    )
    st.markdown("""---""")
    provider_df = preprocess_company_data(provider_df)
    consumer_df = preprocess_company_data(consumer_df)
    uploaded_file = st.file_uploader(
        "Choose a resume file", type=["pdf", "docx", "txt"]
    )
    if uploaded_file is not None:
        resume_text = load_resume(uploaded_file)
        st.write(resume_text[:500] + "...")

    if uploaded_file is not None and not provider_df.empty:
        company_embeddings = generate_embeddings(
            model, provider_df["combined_info"].tolist()
        )
        resume_embedding = generate_embeddings(model, [resume_text])[0]

        similarities = calculate_similarities(resume_embedding, company_embeddings)
        provider_df["Similarity"] = similarities * 100
        sorted_companies = provider_df.sort_values("Similarity", ascending=False)

        display_df = sorted_companies[
            ["Company Name", "Similarity", "Contact Information", "Basic Company Information"]
        ].copy()
        display_df["Similarity"] = display_df["Similarity"].apply(lambda x: f"{x:.2f}%")
        display_df = display_df.reset_index(drop=True)
        display_df.index += 1

        st.subheader("Matches:")
        num_matches = st.slider(
            "Number of matches to display",
            min_value=1,
            max_value=len(sorted_companies),
            value=10,
        )
        st.table(display_df.head(num_matches))

with provider_df_tab:
    st.dataframe(provider_df, height=1000, use_container_width=True)

with consumer_df_tab:
    st.dataframe(consumer_df, height=1000, use_container_width=True)
