import argparse
import hashlib
import io
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from pinecone import init, Index
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class RAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', llm_engine='gpt-3.5-turbo', top_k=3, search_threshold=0.8, max_token_length=512, verbose=False):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        self.llm_api_key = os.getenv('OPENAI_API_KEY')
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.search_threshold = search_threshold
        self.max_token_length = max_token_length
        self.verbose = verbose
        self.model = SentenceTransformer(embedding_model)
        self.llm_engine = llm_engine

        # Initialize Pinecone client
        init(api_key=self.pinecone_api_key)
        self.index = Index(self.pinecone_index_name)

    # The rest of your class definition remains unchanged...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load text data from a JSON file and process with RAG.")
    parser.add_argument("--json_file", default="default_data.json", help="Path to the JSON file containing text data.")
    args = parser.parse_args()

    # Initialize your RAG instance
    rag = RAG()

    # Load and process the specified JSON file
    rag.load_and_process_json(args.json_file)
