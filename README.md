URL Summarizer with LangChain and Groq
A Streamlit web app that summarizes content from YouTube videos or websites using LangChain and Groq's LLM (Gemma2-9b-It).

Features
Extracts and summarizes text from YouTube URLs or web pages

Uses Groq's fast inference API for quick results

Customizable prompt template for summarization

Handles NLTK data dependencies with fallback options

Setup
Add your Groq API key in the sidebar

Enter a valid YouTube or website URL

Click "Summarize" to generate a 1000-word summary

Dependencies: streamlit, langchain-groq, nltk, validators

Note: Requires internet connection for URL processing and Groq API access.
