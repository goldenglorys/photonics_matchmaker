<h1 align="center">
🔬 Photonics Matchmaker
</h1>

<p align="center">
Connecting professionals with companies in the photonics industry through AI-powered matching.
</p>

## 📖 About

Photonics Matchmaker is a Streamlit application that uses advanced language models and embedding techniques to match professionals with companies in the photonics industry. It provides data-driven recommendations and detailed analyses of potential matches.

## 🔧 Features

- Upload resumes 📁 (PDF, DOCX, TXT) and match them with company profiles.
- Use embedding models for initial matching 🧮.
- Provide detailed analysis of matches using language models 🤖.
- Customizable number of matches to analyze 🔢.
- FAQ section for user guidance ❓.

## 💻 Running Locally

1. Clone the repository 📂
```bash
https://github.com/goldenglorys/photonics_matchmaker
cd photonics_matchmaker
```

2. Install dependencies with [Poetry](https://python-poetry.org/) and activate virtual environment 🔨

```bash
poetry install
poetry shell
```

3. Set up environment variables 🔐
Create a .streamlit/config.toml file in the root directory and add the following variables:

```bash
GROQ_API_KEY=""
EMBEDDING_MODEL=""
[server]
maxUploadSize = 25
[browser]
gatherUsageStats = false
[connections.gsheets]

```

4. Run the Streamlit server🚀

```bash
cd photonics_matchmaker
streamlit run main.py
```

## Customization

You can increase the max upload file size by changing `maxUploadSize` in `.streamlit/config.toml`.
Currently, the max upload size is 25MB for the hosted version.

## 📁 Project Structure

```bash
photonics_matchmaker/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── config.py
│
├── components/
│   ├── __init__.py
│   ├── sidebar.py
│   └── faq.py
│
├── core/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── embedding.py
│   ├── language_model.py
│   ├── keyword_extraction.py
│   └── text_summarization.py
│
├── utils/
│   ├── __init__.py
│   └── helpers.py
│
├── data/
│   ├── pro_resume_1.txt
│   └── student_resume_1.txt
│
├── tests/
│   └── __init__.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Upcoming Features

- Add support for more resume formats (e.g., LinkedIn profiles 🔗, online portfolios 🌐)
- Implement a feedback system for improving match quality 📈
- Integrate with job posting APIs for real-time opportunities 🔄
- Develop a company-facing interface for posting job openings 💼
- Implement advanced analytics for industry trends and skills demand 📊

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is MIT licensed.

## 🙏 Acknowledgements

- Streamlit for the amazing web app framework
- Sentence Transformers for the embedding models
- Groq for the language model API
- All the contributors who have helped shape this project