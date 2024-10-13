HF_TOKEN = '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'


import streamlit as st
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import qdrant_client
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import PromptTemplate

# Load the documents
documents = SimpleDirectoryReader("data").load_data()

# Setup the embedding model and LLM
Settings.embed_model = FastEmbedEmbedding()
Settings.llm = HuggingFaceInferenceAPI(
    model_name='mistralai/Mistral-7B-Instruct-v0.3',
    token=HF_TOKEN,
)

# Initialize Qdrant vector store
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name='quickstart')
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine()

# Streamlit app interface
st.title('🦙 Llama Banker')
user_query = st.text_input('Input your prompt here')

if user_query:
    # Retrieve context and prepare the query prompt
    context = query_engine.retrieve(user_query)
    context_data = [info.text for info in context]

    template = (
        '''
        <s>[INST]
        You are a helpful assistant. Use the following pieces of Context to answer the question.
        If you don't know the answer, just say you don't know, don't try to make up an answer.
        {context_str}
        {query_str}
        [/INST]
        '''
    )
    prompt = PromptTemplate(template=template)
    query_engine.update_prompts({'response_synthesizer:text_qa_template': prompt})

    # Get the response from the query engine
    response = query_engine.query(user_query)
    
    # Display only the main response (without metadata)
    st.write(response.response)

