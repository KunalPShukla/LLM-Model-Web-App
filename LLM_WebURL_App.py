#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import libraries
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import streamlit as st
import trafilatura

# Function to process URLs and generate answers
def process_urls_and_generate_answer(urls, search_query):
    # Extract content from all URLs
    comb_text = ''
    for i in urls:
        downloaded = trafilatura.fetch_url(i)
        text = trafilatura.extract(downloaded)
        comb_text = comb_text + text

    # Create a document from the combined text
    from langchain.docstore.document import Document
    document = Document(page_content=comb_text)

    # Split text into chunks
    r_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n'],
        chunk_size=150,
        chunk_overlap=25)
    chunks = r_splitter.split_documents([document])

    # Encode text into vectors
    encode = SentenceTransformer('all-mpnet-base-v2')
    texts = [chunk.page_content for chunk in chunks]
    vectors = encode.encode(texts)

    # Create a FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Encode the search query
    vec = encode.encode(search_query)
    search_vec = np.array(vec).reshape(1, -1)

    # Search for relevant chunks
    dist, indices = index.search(search_vec, k=4)
    flattened_indices = indices.flatten()
    flattened_indices = [int(i) for i in flattened_indices]
    relevant_chunks = [chunks[i] for i in flattened_indices]
    relevant_text = " ".join([chunk.page_content for chunk in relevant_chunks])

    # Load a pre-trained model and tokenizer
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(relevant_text, return_tensors="pt", max_length=256, truncation=True)

    # Generate the answer
    output = model.generate(input_ids=input_ids)

    # Decode the generated text
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

# Streamlit app
def main():
    st.title("LLM Model Web App")
    st.write("Enter a list of URLs and a search query to get an answer.")

    # Input for URLs
    urls = st.text_area("Enter URLs (one per line)", height=150)
    urls = urls.split('\n')

    # Input for search query
    search_query = st.text_input("Enter your search query")

    if st.button("Get Answer"):
        if urls and search_query:
            with st.spinner("Processing..."):
                answer = process_urls_and_generate_answer(urls, search_query)
                st.success("Answer generated!")
                st.write("Question:", search_query)
                st.write("Generated Answer:", answer)
        else:
            st.error("Please provide both URLs and a search query.")

if __name__ == "__main__":
    main()


# In[ ]:




