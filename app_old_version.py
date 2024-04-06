import streamlit as st
from PyPDF2 import PdfReader
import os, re, io
import tempfile
from dotenv import load_dotenv
from pdf2image import convert_from_path
import fitz
from collections import OrderedDict
from my_rag import RAG

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
pdf_storage_dir = 'data/pdfs'
chunk_size = 500 # 500 characters default
chunk_overlap = 25 # default
top_k = 3 # number of chunks found in semantic search
DBPATH = 'data/db_file.db'
model = 'all-MiniLM-L6-v2'

# instantiate RAG class
st_rag = RAG(db_path = DBPATH, llm_api_key=OPENAI_API_KEY, embedding_model=model, chunk_size = chunk_size, overlap=chunk_overlap, top_k = top_k)


# Get the image of the source page
def get_page_image(pdf_filename, page_number):
    pdf_path = os.path.join(pdf_storage_dir, pdf_filename)
    
    try:
        pdf = fitz.open(pdf_path)
        page = pdf.load_page(page_number)

        # Convert the page to an image
        page_image = page.get_pixmap()

        # Get image data in PNG format and write to BytesIO
        image_bytes = io.BytesIO(page_image.tobytes("png"))
        
        return image_bytes
    except Exception as e:
        st.error(f"Error loading page image: {e}")
        return None
    

# Clear the conversation, removing the history
def clear_conversation():
    st.session_state['conversation_history'] = []
    st.session_state['new_query'] = ''
    st.rerun()

# Main UI function
def run_UI():
    st.set_page_config(page_title="Chat with Docs", layout="wide")
    st.header("Chat with Docs: Interact with Your Documents")
    st.write("This app will allow you to interact with the documents in the database. If you have not already done so, please add documents using the sidebar panel.")


    # Initialize session states
    # Assuming each item in conversation_history is a dictionary with keys:
    # 'query', 'rag_response', 'llm_response', 'source_pages'   
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []


    # Clear conversation button
    if st.button("Clear Conversation"):
        clear_conversation()

    # Display the conversation history
    # Display the conversation history
    for i, convo in enumerate(st.session_state['conversation_history']):
        st.text_area(f"Question {i+1}", value=convo['query'], height=75, disabled=True)
        col1, col2 = st.columns(2)
        with col1:
            st.text_area(f"RAG Response {i+1}", value=convo['rag_response'], height=250, disabled=True)
        with col2:
            st.text_area(f"LLM Response {i+1}", value=convo['llm_response'], height=250, disabled=True)
        
        # Show source page buttons
        for j, (pdf_filename, page_number) in enumerate(convo['source_pages']):
            if st.button(f"Show source for Q{i+1} - File {pdf_filename} - Page {page_number + 1}", key=f'source_button_{i}_{j}'):
                image_bytes = get_page_image(pdf_filename, page_number)
                st.image(image_bytes, caption=f"Source: {pdf_filename} (Page {page_number + 1})")

        st.markdown("---")  # Horizontal line

    # Input and button for new query
    new_query = st.text_input("Ask a question", key='new_query')
    

    # On submitting a new query
    if st.button("Submit") and new_query:
        # Concatenate previous conversation with new query
        full_conversation = '\n'.join([item['query'] + '\n' + item['rag_response'] for item in st.session_state['conversation_history']])
        full_conversation += '\nQ: ' + new_query

        # Generate responses
        rag_response = 'A: ' + st_rag.generate_response(full_conversation)
        llm_response = 'A: ' + st_rag.integrate_llm(full_conversation)
        #print(full_conversation)

        # Get info for the source page display
        chunk_ids = st_rag.semantic_search(new_query)
        best_chunks_and_references = st_rag.get_chunks_by_ids(chunk_ids)
        current_source_pages = [(ref[0], ref[1]) for _, ref in best_chunks_and_references]

        #Update conversation history
        st.session_state['conversation_history'].append({
        'query': 'Q: ' + new_query,
        'rag_response': rag_response,
        'llm_response': llm_response,
        'source_pages': current_source_pages
    })

        # Clear the input field and rerun
        #st.session_state['new_query'] = ''
        st.rerun()


    # Sidebar menu
    with st.sidebar:
        st.subheader("Settings")

        # Sliders for chunk size and overlap
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 500
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 25

        st.session_state.chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=st.session_state.chunk_size)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=100, value=st.session_state.chunk_overlap)

        # Button to clear the database
        if st.button("Clear Database"):
            st_rag.clear_database()
            st.write("Database cleared.")

    with st.sidebar:
        st.subheader("Document Uploader")

        # Document uploader
        pdf_files = st.file_uploader("Upload documents", type="pdf", key="upload", accept_multiple_files=True)

        # Process the document after the user clicks the button
        if st.button("Process Files"):
            with st.spinner("Processing"):
                text_chunks = st_rag.extract_and_store_text(pdf_files)
              
                for pdf_file in pdf_files:
                    # Define the path to save the PDF
                    file_path = os.path.join(pdf_storage_dir, pdf_file.name)

                    # Write the contents of the uploaded file to a new file
                    with open(file_path, "wb") as f:
                        f.write(pdf_file.getbuffer())

                    st.write(f"Saved {pdf_file.name} to {pdf_storage_dir}")
 
            

# Application entry point
if __name__ == "__main__":
    # Run the UI
    run_UI()
