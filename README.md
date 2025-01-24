# LLM Web URL App

A Streamlit web application that processes a list of URLs and generates answers to user queries using advanced natural language processing (NLP) techniques. Built with LangChain, Sentence Transformers, FAISS, and T5-base, this app is perfect for extracting and summarizing information from multiple web sources.

## Features

- **URL Processing**: Extracts and combines text content from multiple URLs.
- **Query-Based Answers**: Generates answers to user queries using a T5-base model.
- **FAISS Indexing**: Efficiently searches for relevant text chunks using FAISS.
- **Streamlit Interface**: User-friendly web interface for easy interaction.

## How It Works

1. Enter a list of URLs (one per line) and a search query.
2. The app extracts text from the URLs, splits it into chunks, and encodes it into vectors using Sentence Transformers.
3. A FAISS index is created to search for the most relevant text chunks.
4. The T5-base model generates an answer based on the relevant text.
