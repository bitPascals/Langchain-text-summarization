import os
import nltk
import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


try:
    nltk.download('punkt', timeout=60)
    nltk.download('averaged_perceptron_tagger', timeout=60)
except:
    # If online download fails, use the manual method
    print("Online download failed. Please download manually:")
    print("1. Go to https://www.nltk.org/nltk_data/")
    print("2. Download 'punkt' and 'averaged_perceptron_tagger' packages")
    print("3. Place them in one of these folders:")
    for path in nltk.data.path:
        print(f"- {path}")

## Streamlit app setup
st.set_page_config(page_title="Langchain: Summarize texts from Youtube and Websites", page_icon="🐦")
st.title("🐦Langchain: Summarize texts from Youtube and Websites")
st.subheader("Summarize URL")

# Get the Groq API Key and (Youtub or website) URL to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter the  URL to be summarized", label_visibility="collapsed", placeholder="https://wwww.example.com")

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

prompt_template = """
Provide the summary of the following text in 1000 words:
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("Summarize"):
    # Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please provide a valide Youtube or Website URL")    
       
    else: 
        try:
            with st.spinner("Loading..."):
                # Loading the youtube or website data
                if "youtube.com" in generic_url:
                    loader = WebBaseLoader(generic_url)
                    docs = loader.load()
                else: 
                    loader = UnstructuredURLLoader(urls=[generic_url], 
                    ssl_verify=False, 
                    headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                        "(KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            "Accept-Language": "en-US,en;q=0.5",
                            "Connection": "keep-alive"
                            }
                    )
                docs =  loader.load()
                st.success("Data loaded successfully")

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt)
                output_summary = chain.run({"input_documents": docs})

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")