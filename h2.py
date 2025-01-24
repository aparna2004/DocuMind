import os
import tempfile
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub

load_dotenv()
# Set up Hugging Face token (make sure to replace with your actual token)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit layout setup
st.title("Document Question Answering System")
st.write("Welcome to the Document Question Answering System built using Langchain and Hugging Face.")

# Step 1: Upload PDF document(s)
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_documents = []
    for uploaded_file in uploaded_files:
        # Create a temporary file and write the uploaded content to it
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        # Load the uploaded PDF file from the temporary path
        loader = PyPDFLoader(tmp_file_path)
        document = loader.load()
        all_documents.extend(document)
    
    # Step 2: Split documents into chunks
    st.write("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_documents)
    st.write(f"Total document chunks: {len(final_documents)}")

    # Step 3: Generate embeddings using Hugging Face
    st.write("Generating embeddings...")
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Example embedding output for verification
    example_embedding = np.array(huggingface_embeddings.embed_query(final_documents[0].page_content))
    st.write("Example embedding shape:", example_embedding.shape)

    # Step 4: Create and persist the ChromaDB vector store
    st.write("Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=final_documents,
        embedding=huggingface_embeddings,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    st.write("ChromaDB vector store created and persisted.")

    # Step 5: Perform similarity search
    query = st.text_input("Enter your query to search documents:")
    if query:
        relevant_documents = vectorstore.similarity_search(query)
        st.write("Relevant document content:", relevant_documents[0].page_content)

        # Step 6: Set up retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        st.write("Retriever initialized.")

        # Step 7: Configure Hugging Face Hub
        hf = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-v0.1",
            model_kwargs={"temperature": 0.1, "max_length": 500}
        )

        # Test the model invocation
        if st.button("Invoke Model"):
            response = hf.invoke(query)
            st.write("Response from model:", response)

        # Step 8: Define a custom prompt template
        prompt_template = """
        Use the following piece of context to answer the question asked.
        Please try to provide the answer only based on the context

        {context}
        Question: {question}

        Helpful Answers:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Step 9: Set up RetrievalQA
        retrievalQA = RetrievalQA.from_chain_type(
            llm=hf,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Step 10: Query the QA system
        if query:
            result = retrievalQA.invoke({"query": query})
            st.write("Result:", result['result'])

else:
    st.write("Please upload one or more PDF documents to begin.")
