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

    def process_text(self, source, text, chunk_id):
        unique_id = hashlib.sha256(f"{source}_{chunk_id}".encode()).hexdigest()
        embedding = self.model.encode(text)
        self.index.upsert(id=unique_id, vectors=[embedding], metadata={"source": source, "text": text})

    def load_and_process_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for source, text in data.items():
            if isinstance(text, list):
                for i, chunk in enumerate(text):
                    self.process_text(source, chunk, i)
            else:
                self.process_text(source, text, 0)

    def semantic_search(self, query):
        query_embedding = self.model.encode(query)
        results = self.index.query(queries=[query_embedding], top_k=self.top_k)
        return [match["id"] for match in results["matches"]]

    def generate_response(self, query):
        best_chunk_ids = self.semantic_search(query)
        if best_chunk_ids:
            responses = []
            for best_chunk_id in best_chunk_ids:
                metadata = self.index.fetch_metadata(ids=[best_chunk_id]).get("metadata", {})
                if metadata:
                    responses.append(metadata[0]["text"])
            combined_chunks = " ".join(responses)
            return self.integrate_llm(combined_chunks + "\n" + query)
        else:
            return "Sorry, I couldn't find a relevant response."

    def integrate_llm(self, prompt):
        try:
            response = openai.ChatCompletion.create(model=self.llm_engine, messages=[{"role": "system", "content": "You are an expert in this content, helping to explain the text"}, {"role": "user", "content": prompt}])
            return response.choices[0].message['content']
        except Exception as e:
            print(f"Error in generating response: {e}")
            return None

# Example usage with command-line argument for specifying the JSON file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load text data from a JSON file and process with RAG.")
    parser.add_argument("--json_file", default="default_data.json", help="Path to the JSON file containing text data.")
    args = parser.parse_args()

    # Initialize your RAG instance 
    rag = RAG(pinecone_api_key="your_pinecone_api_key", pinecone_index_name="your_index_name", llm_api_key="your_llm_api_key")

    # Load and process the specified JSON file
    rag.load_and_process_json(args.json_file)
