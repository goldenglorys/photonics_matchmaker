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
            # FAQ: The Hitchhiker's Guide to Photonic Matchmaking

            ## How does the embedding model mechanisms work?

            Imagine if traditional matchmaking service used quantum entanglement instead of swipes. 
            That's kind of what we're doing here, but for photonics professionals and companies.
            
            By analyzing the semantic similarity between job seekers' resumes and company profiles,
            the system provides data-driven recommendations to streamline the job matching process.

            Our system is like a nerdy cupid with a Ph.D. in semantics:

            1. Data Ingestion: 
                We slurp up company data from Google Sheets (because who doesn't love a good spreadsheet?)
                and professionals upload their resumes (we accept PDF, DOCX, TXT, and interpretive dance videos*).

            2. Text Preprocessing: 
                We blend company info into a smooth, digestible text smoothie. Resumes go through our
                SOTA "jargon-to-English" translator.

            3. Semantic Embedding: 
                We use a pre-trained Sentence Transformer model to turn text into high-dimensional vectors. 
                It's like giving words superpowers and letting them fly in semantic space.

            4. Similarity Computation: 
                We play matchmaker using cosine similarity. It's like comparing the angle between two vectors, 
                but way less boring than it sounds.

            5. Results Presentation: 
                We rank companies based on similarity scores,  presented as percentages. It's like a beauty pageant, 
                but for job matches, and without the swimsuit competition.  Users can adjust the number of matches displayed, 
                allowing for comprehensive or focused views.

            `Just kidding about the dance videos. Please don't send those.`
            
            ---

            ## What's this about keyword extraction and text summarization?

            Ah, you've stumbled upon our secret sauce! We use some fancy AI tricks to make sense of all that text:

            1. Keyword Extraction:
                Imagine if AI could read a whole book and tell you the most important words on a Post-it note. 
                That's KeyBERT! We use it to extract the crucial keywords from company profiles and resumes. 
                It's like having a super-smart highlighter that knows exactly which words matter most in the photonics world.

                For example, it might pick out words like "laser", "optics", "photolithography", or "quantum computing" 
                from a sea of text. It's particularly useful for those long-winded company descriptions that sound like 
                they were written by a committee of very enthusiastic engineers.

            2. Text Summarization:
                Meet Sumy, our AI cliff-notes generator. It takes those long, rambling paragraphs and turns them into concise, 
                meaningful summaries. It's like having a friend who's really good at explaining movies in one sentence.

                Sumy helps us condense verbose company profiles or extensive resumes into digestible chunks. This way, 
                our matching algorithm doesn't get lost in a forest of words, but instead focuses on the key points.

            Together, these tools help us create a more accurate and efficient matching process. It's like giving our AI system 
            a pair of super-powered reading glasses and a knack for getting to the point.

            ---

            ## How does the language model mechanism work?
            Our current system is like a really smart library assistant who's really good at finding relevant book passages 
            but sometimes gets a bit creative with their summaries.

            Here's how it works:

            1. Chunky Goodness: We break your document into bite-sized chunks, like a literary pizza.
            2. Vector Indexing: We create a special index of these chunks, kind of like a map of where all the good stuff is hidden.
            3. Semantic Search: When you ask a question, we use this map to find the most relevant chunks.
            4. Answer Generation: We feed these chunks to our AI language model, which is like a hyper-intelligent space octopus that's 
            really good at understanding context and generating human-like text.

            The language model approach is like giving this system a turbo boost. 
            Instead of just finding and regurgitating relevant chunks, it could:

            1. Synthesize information from multiple chunks
            2. Infer answers that aren't explicitly stated in the text
            3. Provide more nuanced and context-aware responses
            4. Potentially handle more complex, open-ended questions

            It's like upgrading from a library assistant to a panel of Nobel laureates who've read your entire document 
            and can discuss it in depth. But remember, even Nobel laureates can sometimes say silly things after too much coffee.

            **TLDR;** When you upload a document, it will be divided into smaller chunks 
            and stored in a special type of database called a vector index 
            that allows for semantic search and retrieval.

            When you ask a question, (in this case, match-make) it will search through the
            document chunks and find the most relevant ones using the vector index.
            Then, it will use ML model reasoning to generate a final answer.

            ---

            ## Is my data safe?
            Yes, your data is safe. The app does not store your documents or
            questions. All uploaded data is deleted after you close the browser tab.

            ---

            ## Are the answers 100% accurate?
            Is anything in life 100% accurate? Our AI is smart, but it's not infallible. 
            It's like that friend who always has an answer but sometimes confuses Star Wars with Star Trek. 
            Always double-check crucial information!

            But for most use cases, it is very accurate. Always check with your intuition to make sure that the results
            are correct.

            ---

            ## How can I improve my matches?

            1. Use industry-specific keywords in your resume. Our AI loves buzzwords almost as much as VCs do.
            2. Be clear and concise. Our AI is smart, but it doesn't have time to read your life story.
            3. Update your skills regularly. The photonics world moves fast – make sure your resume keeps up!

            ---

            ## What if I get matched with my ex-boss?

            Well, that's awkward. But look on the bright side – at least you know you're qualified for the job! 
            Maybe it's time to bury the hatchet... or find a really good disguise.

            ---

            ## How can I make this system better?

            We're always looking for ways to improve! Here are a few ideas:

            1. Feed it more data: The more resumes and company profiles we have, the smarter it gets. 
            It's like sending it to a really nerdy school.
            2. Give feedback: Let us know when it makes a great match or when it's way off. 
            It's like training a puppy, but with less drool and more data.
            3. Suggest new features: Maybe you want it to match based on your favorite type of laser or your 
            stance on the wave-particle duality debate. Let us know!

            Remember, in the world of photonics matchmaking, we're all just trying to find that perfect wavelength. 
            Stay bright, stay coherent, and may the photons be with you!
        """
    )
