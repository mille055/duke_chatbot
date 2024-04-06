import streamlit as st
from rag import RAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG with environment variables or directly with your keys
rag = RAG()

# Set Streamlit page configuration to match the desired style
st.set_page_config(page_title="Duke AIPI Chatbot", layout="wide")

# Main UI function
def run_UI():
    # Page styling to match the example (as closely as possible with Streamlit)
    st.markdown("""
        <style>
        .stApp { background-color: #fafafa; }
        </style>
    """, unsafe_allow_html=True)

    st.header("Duke AIPI Chatbot")
    st.write("What questions do you have about the program?")

    # Initialize session states for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # Input field for new query
    new_query = st.text_input("Ask a question", key='new_query')

    # Submit button for new query
    if st.button("Submit") and new_query:
        # Generate response using RAG
        response = rag.generate_response(new_query)

        # Update conversation history
        st.session_state['conversation_history'].append({'query': new_query, 'response': response})

        # Clear the input field for next query
        st.session_state['new_query'] = ''
        st.experimental_rerun()

    # Display the conversation history
    for i, convo in enumerate(st.session_state['conversation_history']):
        st.text_area(f"Question {i+1}", value=convo['query'], height=75, disabled=True)
        st.text_area(f"Response {i+1}", value=convo['response'], height=150, disabled=True)
        st.markdown("---")  # Horizontal line for separation

if __name__ == "__main__":
    run_UI()
