HF_TOKEN = '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'

import os
import streamlit as st
from llama_index.core import Document
from llama_index.core import Settings
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import PromptTemplate
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

# Function to extract text from PDF with caching
@st.cache_data
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Load PDF content and store it as Document objects with caching
@st.cache_resource
def load_documents():
    return SimpleDirectoryReader("data").load_data()

documents = load_documents()  # Wrap text in Document objects

# Setup embedding model and LLM with caching

Settings.embed_model = FastEmbedEmbedding()
Settings.llm = HuggingFaceInferenceAPI(
    model_name='mistralai/Mistral-7B-Instruct-v0.3',
    token=HF_TOKEN,
)

# Initialize Qdrant vector store
@st.cache_resource
def init_vector_store():
    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name='financial_strategies')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

storage_context = init_vector_store()

# Build and cache the index
@st.cache_resource
def build_index(_documents, _storage_context):  # Note the underscore in both arguments
    index = VectorStoreIndex.from_documents(_documents, storage_context=_storage_context)
    return index

index = build_index(documents, storage_context)
query_engine = index.as_query_engine()

# Streamlit app interface improvements

# Header and Title with custom styles
st.markdown("<h1 style='text-align: center;'>💼 ClarityX:Transforming Business with AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get actionable financial strategies from industry-standard documents</p>", unsafe_allow_html=True)

# Sidebar with instructions
st.sidebar.title("How to Use")
st.sidebar.write("""
1. Enter a business question related to finance strategy.
2. The system retrieves relevant financial insights and generates a custom strategy.
3. View context and strategy along with any relevant data insights.
""")

# Input field for user query
user_query = st.text_input('🔍 Ask a Financial Strategy Question:')

# Add a submit button
if st.button('Get Strategy'):
    if user_query:
        with st.spinner('Retrieving and generating strategy...'):
            # Retrieve context from stored documents based on user query
            context = query_engine.retrieve(user_query)  # Limit number of results
            context_data = [info.text for info in context]

            # Prepare context for display in the response
            context_str = '\n'.join(context_data)

            # Define prompt template for context-aware response
            template = (
                '''
                <s>[INST]
                You are a business finance assistant. Based on the following context, provide a detailed financial strategy.
                {context_str}
                {query_str}
                [/INST]
                '''
            )

            # Generate prompt and query
            prompt = PromptTemplate(template=template)
            query_engine.update_prompts({'response_synthesizer:text_qa_template': prompt})
            response = query_engine.query(user_query)

            # Display results in tabs for better organization
            tab1, = st.tabs(["📋 Financial Strategy"])

            # Tab 1: Show the strategy response
            with tab1:
                st.subheader('📋 Financial Strategy Response:')
                st.write(response.response)

            # Tab 2: Show relevant data insights if applicable
            # with tab2:
            #     st.subheader("📊 Financial Data Insights (If Available)")
            #     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

            #     if uploaded_file is not None:
            #         try:
            #             # Load CSV data
            #             df = pd.read_csv(uploaded_file)
            #             st.write("Data Preview:")
            #             st.dataframe(df.head())  # Display first few rows
            #             # Visualize the data as a line chart
            #             # st.line_chart(df)
            #         except Exception as e:
            #             st.error(f"Error loading CSV file: {e}")
            #     else:
            #         st.write("Upload a CSV file to visualize data insights.")
    else:
        st.warning("Please enter a valid business question.")

# Footer with information
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 ClarityX - Financial Strategy Assistant</p>", unsafe_allow_html=True)
