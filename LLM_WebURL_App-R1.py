#!/usr/bin/env python
# coding: utf-8

# In[61]:


# Install necessary libraries
# !pip install langchain sentence-transformers faiss-cpu transformers trafilatura


# In[60]:


# Import libraries
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import trafilatura

# Function to process URLs and generate answers
def process_urls_and_generate_answer(urls, search_query):
    # Extract content from all URLs
    comb_text = ''
    for url in urls:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        if text:  # Check if text is not None
            comb_text += text

    # Create a document from the combined text
    from langchain.docstore.document import Document
    document = Document(page_content=comb_text)

    # Split text into chunks
    r_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n'],
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = r_splitter.split_documents([document])

    # Encode text chunks into embeddings
    encode = SentenceTransformer('gtr-t5-large')  # Using gtr-t5-large for embeddings
    texts = [chunk.page_content for chunk in chunks]
    vectors = encode.encode(texts)

    # Create a FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Define the search query
    search_query = "Which players are performing good?"

    # Encode the search query
    vec = encode.encode(search_query)
    search_vec = np.array(vec).reshape(1, -1)

    # Search for relevant chunks
    dist, indices = index.search(search_vec, k=4)
    flattened_indices = indices.flatten()
    flattened_indices = [int(i) for i in flattened_indices]
    relevant_chunks = [chunks[i] for i in flattened_indices]

    # Load a pre-trained model and tokenizer
    model_name = "google/flan-t5-large"  # Upgraded model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Wrap the model in a HuggingFacePipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Define the prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    # Create the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Load the QA chain
    qa_chain = load_qa_chain(
        llm=llm,  # Use the wrapped HuggingFacePipeline
        chain_type="stuff",  # Use "stuff" for small contexts
        prompt=prompt  # Use the defined prompt template
    )

    # Generate the answer
    answer = qa_chain.run(input_documents=relevant_chunks, question=search_query)

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




