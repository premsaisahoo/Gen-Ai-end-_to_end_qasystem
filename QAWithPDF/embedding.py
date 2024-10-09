from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.core import ServiceContext

import os
from dotenv import load_dotenv
load_dotenv()


import sys
from exception import customexception
from logger import logging


google_api_key = os.getenv("GOOGLE_API_KEY")

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        Settings.llm = Gemini(models = "gemini-pro",api_key = google_api_key)
        Settings.embed_model = GeminiEmbedding(model="models/embedding-001")
        Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)

       

        #service_context = ServiceContext.from_defaults(llm=model,embed_model=gemini_embed_model, chunk_size=800, chunk_overlap=20)
        
        logging.info("")
        index = VectorStoreIndex.from_documents(document, embed_model=gemini_embed_model)

        #index = VectorStoreIndex.from_documents(document,service_context=service_context)
        index.storage_context.persist()
        
        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise customexception(e,sys)