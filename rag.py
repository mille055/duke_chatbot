import argparse
import hashlib
import io
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from pinecone import init, Index, Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import logging

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
        #self.tokenizer = AutoTokenizer.from_pretrained("YourFineTunedModel")
        #self.model = AutoModelForCausalLM.from_pretrained("YourFineTunedModel")
        
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.pinecone_index_name)
    

        # Create the Pinecone store
    def create_pinecone(self, json_file, index_name='newindex'):
            
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
            #self.index = self.pc.Index(self.pinecone_index_name)
        self.pc.create_index(
                name=self.pinecone_index_name,
                dimension= 384, #1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                    )
                )
        self.index = self.pc.Index(self.pinecone_index_name)
        self.load_and_process_json(json_file)


    def process_text(self, source, text, chunk_id):
        unique_id = hashlib.sha256(f"{source}_{chunk_id}".encode()).hexdigest()
        embedding = self.model.encode(text).tolist()
        print('type of embedding is ', type(embedding))
        print(unique_id, embedding, source)
        #self.index.upsert(id=unique_id, vectors=embedding, metadata={"source": source, "text": text})
        self.index.upsert(vectors=[{"id": unique_id, "values":embedding, "metadata":{"source": source, "text": text}}])

    def load_and_process_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            

        for source, text in data.items():
            print(f"Processing text: {source}")
            if isinstance(text, list):
                for i, chunk in enumerate(text):
                    print(i, chunk)
                    self.process_text(source, chunk, i)
            
            else:
                 print(text[0:40])
                 self.process_text(source, text, 0)


    def semantic_search(self, query):
        source_list = []
        texts = []
        try:
            query_embedding = self.model.encode(query).tolist()
            results = self.index.query(vector=query_embedding, top_k=self.top_k, include_metadata=True)
            matches = [match for match in results["matches"]]
            for match in matches:
                source_list.append(match['metadata']['source'])
                texts.append(match['metadata']['text'])
            return texts, source_list
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return [], []
        

    def generate_response(self, query):
        
        texts, sources = self.semantic_search(query)
        if texts:
            combined_chunks = " ".join(texts)
            return (self.integrate_llm(combined_chunks + "\n" + query), sources)
        else:
            return ("Sorry, I couldn't find a relevant response.", None)

    def integrate_llm(self, prompt):
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            response_ids = self.model.generate(input_ids, max_length=self.max_token_length)
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            return response_text
        except Exception as e:
            print(f"Error in generating response: {e}")
            return "An error occurred while generating a response."

# Example usage with command-line argument for specifying the JSON file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load text data from a JSON file and process with RAG.")
    parser.add_argument("--json_file", default="data/extracted_data_2024-04-01_07-59-36.json", help="Path to the JSON file containing text data.")
    args = parser.parse_args()

    # Initialize your RAG instance 
    rag = RAG()

    # Load and process the specified JSON file
    #rag.create_pinecone(args.json_file, rag.pinecone_index_name)

    # Query the pinecone vector storage
    phrase = 'Submit transcripts and letters of recommendation'
    texts, sources = rag.semantic_search(phrase)
    print(texts)