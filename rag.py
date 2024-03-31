import streamlit as st
from PyPDF2 import PdfReader
import os, re, io
import tempfile
#import nougat
from pdf2image import convert_from_path
import fitz
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import openai
#from haystack.nodes import PreProcessor, PDFToTextConverter

class RAG:
     
    def __init__(self, pinecone_api_key, pinecone_index_name, llm_api_key, embedding_model='all-MiniLM-L6-v2', llm_engine = 'gpt-3.5-turbo', chunk_size=250, overlap=25, top_k = 3, search_threshold=0.8, max_token_length=512, cache_size=1000, verbose=False):
        """
        Initializes the RAG instance with Pinecone client and configurations.

        Args:
            pinecone_api_key (str): API key for Pinecone.
            pinecone_index_name (str): Name of the Pinecone index.
            llm_api_key (str): API key for OpenAI's language model.
            embedding_model (str): Name of the sentence transformer model for embeddings.
            chunk_size (int), overlap (int), top_k (int), search_threshold (float),
            max_token_length (int), cache_size (int), verbose (bool): Various configuration options.
        """
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.llm_api_key = llm_api_key
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.search_threshold = search_threshold
        self.max_token_length = max_token_length
        self.cache_size = cache_size
        self.text_dict = {}
        self.verbose = verbose
        self.model = SentenceTransformer(embedding_model)
        self.llm_engine = llm_engine

        # Initialize Pinecone client
        pinecone.init(api_key=self.pinecone_api_key)
        self.index = pinecone.Index(self.pinecone_index_name)

     def store_chunks(self, chunks):
        """
        Stores the processed text chunks in the Pinecone index.

        Args:
            chunks (List[Tuple[str, Tuple[str, int]]]): List of text chunks with references.
        """
        for chunk, references in chunks:
            embedding = self.model.encode(chunk)
            pdf_filename, page_number = references[0]
            self.index.upsert(id=pdf_filename + "_" + str(page_number), vectors=embedding, metadata={"chunk": chunk, "page_number": page_number})

     def semantic_search(self, query):
        """
        Performs semantic search to find the most relevant text chunks for a given query.

        Args:
            query (str): User's query string.

        Returns:
            List[int]: List of chunk IDs representing the top search results.
        """
        query_embedding = self.model.encode(query)
        results = self.index.query(queries=query_embedding, top_k=self.top_k)
        top_chunk_ids = [match["id"] for match in results["matches"]]
        if self.verbose:
            print('Semantic search returning IDs', top_chunk_ids)
        
        return top_chunk_ids

     
    def initialize_database(self):
        """
        Creates necessary tables in the SQLite database if they don't already exist.
        """
        cursor = self.db.cursor()

        # Create the text_chunks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY,
            chunk TEXT NOT NULL,
            pdf_filename TEXT NOT NULL,
            page_number INTEGER NOT NULL
        )
    ''')

        # Create the embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES text_chunks (id)
            )
        ''')

        self.db.commit()
    

    def clear_database(self):
        """
        Clears all data from the text_chunks and embeddings tables in the database.
        """
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM text_chunks")
        cursor.execute("DELETE FROM embeddings")
        self.db.commit()

    def extract_and_store_text(self, pdf_files):
        """
        Extracts text from PDF files, chunks it, and stores it in the database.
        
        Args:
            pdf_files: List of PDF files to process.
        """

        # Extract text from PDF files
        text_dict = self.get_text(pdf_files)

        # Chunk the extracted text
        chunks = self.chunk_text(text_dict)
        #chunks = self.haystack_text_chunker(text_dict)

        # Store the chunks in the database
        self.store_chunks(chunks)

        # Create embeddings
        self.create_embeddings()

        return chunks

    def get_text(self, pdf_files):
        """
        Function to extract the text from one or more PDF file

        Args:
            pdf_files (files): The PDF files to extract the text from

        Returns:
            text_dict (dict): The text extracted for each page number 
                with references to the source pdf and page
        """
        
        text_dict = {}
        for pdf_file in pdf_files:
            pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
            filename = pdf_file.name
            if self.verbose:
                print("Debug: processing file ", filename)

            for page_num in range(len(pdf)):
                if self.verbose:
                    print("Debug: now on page", page_num)
                page = pdf.load_page(page_num)

                current_page_text = page.get_text().replace('\n', ' ').replace('\u2009', ' ').replace('\xa0', ' ')
                current_page_text = ' '.join(current_page_text.split())


                # Store the temporary file name in the dictionary
                text_dict[(filename, page_num)] = (current_page_text)


        return text_dict


    def chunk_text(self, text_dict):
        """
        Splits text into chunks with a specified maximum length and overlap,
        trying to split at sentence endings when possible.

        Args:
            text_dict (dict): Dictionary with page numbers as keys and text as values.
            max_length (int): Maximum length of each chunk.
            overlap (int): Number of characters to overlap between chunks.

        Returns:
            list of tuples: Each tuple contains (chunk, (filename, page_numbers)), 
                            where `chunk` is the text chunk, 'filename' is the source file and `page_numbers` 
                            is a list of page numbers from which the chunk is derived.
        """
        overlap = self.overlap
        chunks = []
        current_chunk = ""
        current_references = []  # Stores (file_name, page_number) tuples

        for (file_name, page_number), (text) in text_dict.items():
            sentences = re.split(r'(?<=[.!?]) +', text)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    chunks.append((current_chunk, current_references.copy()))
                    current_chunk = current_chunk[-overlap:]
                    current_references = list(set(current_references + [(file_name, page_number)]))
                current_chunk += sentence + ' '
                current_references.append((file_name, page_number))
                current_references = sorted(set(current_references), key=current_references.index)

            if current_chunk:
                chunks.append((current_chunk, current_references))
                current_chunk = ""
                current_references = []

        return chunks

    # Note: tried to use haystack but messed up dependencies with pydantic so
    # could not even try to test this code
    # def haystack_text_chunker(self, text_dict):
        
    #      processor = PreProcessor(
    #          clean_empty_lines=True,
    #          clean_whitespace=True,
    #          clean_header_footer=True,
    #          remove_substrings=None,
    #          split_by="word",
    #          split_length=self.chunk_size,
    #          split_respect_sentence_boundary=True,
    #          split_overlap=self.overlap,
    #          add_page_number=True
    #          )

    #      if self.verbose:
    #          print("Debug: text_dict sent to haystack", text_dict)

    #      haystack_chunks = []

    #      for (file_name, page_number), text in text_dict.items():
    #          # Process each text block with the PreProcessor
    #          processed_chunks = processor.process([{"text": text}])

    #          for chunk in processed_chunks:
    #              # Chunk is a dictionary with 'text'
    #              current_chunk = chunk['text']
    #              # Append chunk along with its file name and page number
    #              haystack_chunks.append((current_chunk, [(file_name, page_number)]))

    #      return haystack_chunks

        
    
    def store_chunks(self, chunks):
        """
        Stores the processed text chunks in the database.

        Args:
            chunks (List[Tuple[str, Tuple[str, int]]]): List of text chunks with references.
        """
        
        cursor = self.db.cursor()
        for chunk, references in chunks:
            
            pdf_filename, page_number = references[0]  
            #if self.verbose:
                #print("Debug: reference", pdf_filename, page_number)
            cursor.execute("INSERT INTO text_chunks (chunk, pdf_filename, page_number) VALUES (?, ?, ?)", 
                        (chunk, pdf_filename, page_number))

        self.db.commit()

    def create_embeddings(self):
        """
        Creates embeddings for each chunk stored in the database using the sentence transformer model.
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT id, chunk FROM text_chunks")
        rows = cursor.fetchall()

        for row in rows:
            try:

                raw_text = row[1]
                cleaned_text = ''.join(char for char in raw_text if ord(char) < 128)
                embedding = self.model.encode(cleaned_text)
                cursor.execute("INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)", (row[0], embedding.tobytes()))

            # Rest of the code...
            except Exception as e:
                print(f"Error encoding text: {row[1]}")
                print("Error message:", e)
                continue  # Skip this row and continue with the next

        self.db.commit()

     def load_and_process_json(self, json_file):
        """
        Loads text data from a JSON file, processes, and stores chunks in Pinecone with source information.
        """
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for source, text in data.items():
            # Assuming 'text' might be a large string or a list of chunks
            # If text is not already chunked, you would chunk it here
            if isinstance(text, list):
                for i, chunk in enumerate(text):
                    self.process_text(source, chunk, i)
            else:
                self.process_text(source, text, 0)

    def semantic_search(self, query):
        """
        Performs semantic search to find the most relevant text chunks for a given query.

        Args:
            query (str): User's query string.

        Returns:
            List[int]: List of chunk IDs representing the top search results.
        """
        cleaned_query = ''.join(char for char in query if ord(char) < 128)
        query_embedding = self.model.encode(cleaned_query)
        cursor = self.db.cursor()
        cursor.execute("SELECT chunk_id, embedding FROM embeddings")
        rows = cursor.fetchall()

        similarities = []  # This will store the list of (chunk_id, similarity score)

        for row in rows:
            chunk_id = row[0]
            embedding = np.frombuffer(row[1], dtype=np.float32)
            sim_score = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))  # Calculate similarity score

            similarities.append((chunk_id, sim_score))  # Append the tuple (chunk_id, similarity score)

        # Sort based on similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the top k chunk IDs
        top_chunk_ids = [chunk_id for chunk_id, _ in similarities[:self.top_k]]
        if self.verbose:
            print('Semantic search returning IDs', top_chunk_ids)
        
        return top_chunk_ids
        
    def get_chunks_by_ids(self, chunk_ids):
        """
        Retrieves text chunks and the associated references by the chunk IDs.

        Args:
            chunk_ids (list of int): The IDs of the chunk to retrieve.

        Returns:
            chunks (list of tuples): A list of tuples containing the text chunk and its reference 
                   (PDF filename, page number).
        """
        chunks = []
        cursor = self.db.cursor()
        
        for chunk_id in chunk_ids:
            cursor.execute("SELECT chunk, pdf_filename, page_number FROM text_chunks WHERE id = ?", (chunk_id,))
            result = cursor.fetchone()

            if result:
                chunk, pdf_filename, page_number = result
                chunks.append((chunk, (pdf_filename, page_number)))
                
            else:
                return None
        return chunks


    def integrate_llm(self, prompt):
        """
        Generates a response using a large language model based on the given prompt.

        Args:
            prompt (str): Prompt string including context and query for the language model.

        Returns:
            str: The generated response from the language model.
        """
        
        message=[{"role": "assistant", "content": "You are an expert in this content, helping to explain the text"}, {"role": "user", "content": prompt}]
        try:
            response = openai.chat.completions.create(
                model=self.llm_engine,  
                messages=message,
                max_tokens=250,  
                temperature=0.1  
            )
            # Extracting the content from the response
            chat_message = response.choices[0].message
            if self.verbose:
                print(chat_message)
            return chat_message.content
    
        except Exception as e:
            print(f"Error in generating response: {e}")
            return None
        

    def generate_response(self, query):
        """
        Generates a response to a given query using semantic search and large language model integration.

        Args:
            query (str): The query string for which a response is required.

        Returns:
            str: Generated response to the query.
        """
        
        best_chunk_ids = self.semantic_search(query)
        if best_chunk_ids:
            chunks = []
            cursor = self.db.cursor()

            for best_chunk_id in best_chunk_ids:
                cursor.execute("SELECT chunk FROM text_chunks WHERE id = ?", (best_chunk_id,))
                chunk = cursor.fetchone()
                if chunk:
                    chunks.append(chunk[0])

            combined_chunks = " ".join(chunks)
            response = self.integrate_llm(combined_chunks + "\n" + query)
            return response
        else:
            return "Sorry, I couldn't find a relevant response."
        
        
    def get_page_image(self, pdf_file, page_num):
        """
        Extracts and returns a specific page image from a PDF file.

        Args:
            pdf_file: The PDF file to extract the image from.
            page_num: The page number to extract the image for.

        Returns:
            BytesIO object containing the image.
        """
        pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
        page = pdf.load_page(page_num)
        page_image = page.get_pixmap()

        image_bytes = io.BytesIO()
        page_image.save(image_bytes, 'png')
        image_bytes.seek(0)

        return image_bytes
