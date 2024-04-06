import streamlit as st
from rag import RAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG with environment variables or directly with your keys
rag = RAG()

# Streamlit page configuration
st.set_page_config(page_title="Duke AIPI Chatbot", layout="wide")

def run_UI():
    # Apply custom CSS for styling message boxes
    st.markdown("""
        <style>
        .stApp { background-color: #fafafa; }
        .stheader { background-color: #0577B1}
        .stchat_message("user") { 
            background-color: #0577B1; 
            padding: 10px; 
            border-radius: 15px; 
            margin: 10px; 
            float: left;
            clear: both;
        }
        .stchat_message("assistant") { 
            background-color: #012169; 
            padding: 10px; 
            border-radius: 15px; 
            margin: 10px;
            float: left;
            clear: both;
        }
        </style>
    """, unsafe_allow_html=True)
    # avatars
    avatar_user = '🤔'
    avatar_assistant = 'assets/duke_d_2.png'
    
    st.image('assets/duke_chapel_blue_with_text.png', caption='Duke University')
    # Initialize or retrieve the conversation history from the session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


    for message in st.session_state.conversation_history:
        
        with st.chat_message(message["role"], avatar = message["avatar"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What questions can I help you with?"):
        st.session_state.conversation_history.append({"role": "user", "content": prompt, "avatar": avatar_user})
        with st.chat_message("user", avatar = avatar_user):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=avatar_assistant):
            
            response_text, sources = rag.generate_response(prompt)
            # Append user query and response to conversation history
            #st.session_state.conversation_history.append({"role": "user", "content": prompt})
            st.session_state.conversation_history.append({"role": "assistant", "content": response_text, "avatar": avatar_assistant})
            response = st.write(response_text)
            if sources:
                st.markdown(f"<div style='text-align: right;'><a href='{sources[0]}' target='_blank'><button style='background-color: #4CAF50; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>Learn More</button></a></div>", unsafe_allow_html=True)

        #st.session_state.conversation_history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    run_UI()