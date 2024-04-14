import streamlit as st
from rag import RAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
use_gpt = False
# Initialize RAG with environment variables or directly with your keys
rag = RAG(use_gpt=use_gpt)

# Streamlit page configuration
st.set_page_config(page_title="Duke AIPI Chatbot", layout="wide")

def bot_message(message):
    st.markdown(f'<div class="bot-message" style="display: flex; padding: 5px;">'
                f'<div style="background-color: #012169; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-left:15px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)

def user_message(message):
    st.markdown(f'<div class="user-message" style="display: flex; padding: 5px;">'
                f'<div style="background-color: #00539B; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-left:15px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)



def run_UI():
    # Apply custom CSS for styling message boxes
    st.markdown("""
        <style>
        .stApp { background-color: #fafafa; }
        .stheader { background-color: #0577B1}
        .user-message { 
            background-color: #00539B; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 15px; 
            float: left;
            clear: both;
        }
        .bot-message { 
            background-color: #012169; 
            padding: 10px; 
            border-radius: 15px; 
            margin: 10px;
            float: left;
            clear: both;
        }
        .stChatInputContainer > div {
                background-color: #E5E5E5;
                border-color: #012169;
                padding: 10px;
                border-radius: 15 px;
                margin: 10px;
                float: left;
                clear: both;
        }
        .stChatInputContainer input:focus {
            border-color: #012169;
        }
        .st-checkbox label span {
            background-color: #012169;
            border-color: #012169;
        }
        </style>
    """, unsafe_allow_html=True)
    # avatars
    avatar_user = 'assets/Blue_question_mark_icon.png'
    avatar_assistant = 'assets/duke_d_2.png'
    
    #st.image('assets/duke_chapel_blue_with_text.png', caption='Duke University')
    st.image('assets/mashup_duke_image_title.png', caption='Duke University')


    # Initialize or retrieve the conversation history from the session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'use_gpt' not in st.session_state:
        st.session_state.use_gpt = use_gpt
    
     # Add a checkbox widget for toggling GPT functionality
    st.session_state.use_gpt = st.checkbox('Use GPT')
    rag.use_gpt = st.session_state.use_gpt


    for message in st.session_state.conversation_history:
        
        with st.chat_message(message["role"], avatar = message["avatar"]):
            #st.markdown(message["content"])
            if message["role"]=="user":
                user_message(message["content"])
            else:
                bot_message(message["content"])


    if prompt := st.chat_input("What questions can I help you with?"):
        st.session_state.conversation_history.append({"role": "user", "content": prompt, "avatar": avatar_user})
        with st.chat_message("user", avatar = avatar_user):
            #st.markdown(prompt)
            user_message(prompt)

        with st.chat_message("assistant", avatar=avatar_assistant):
            whole_prompt = 'Please answer the following query:' + prompt + 'and the following context may be helpful' + " ".join([message['content'] for message in st.session_state.conversation_history])
            #print(whole_prompt)
            # select the model to be used
            rag.use_gpt = st.session_state.use_gpt
            response_text, sources = rag.generate_response(prompt)
            # Append user query and response to conversation history
            #st.session_state.conversation_history.append({"role": "user", "content": prompt})
            st.session_state.conversation_history.append({"role": "assistant", "content": response_text, "avatar": avatar_assistant})
            #response = st.write(response_text)
            bot_message(response_text)
            if sources:
                st.markdown(f"<div style='text-align: right;'><a href='{sources[0]}' target='_blank'><button style='background-color: #3F7D7B; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>View Source</button></a></div>", unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: right;'><a href='https://github.com/mille055/duke_chatbot/tree/main/assets/AIPI-Incoming-Student-FAQ.docx' target='_blank'><button style='background-color: #3F7D7B; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>View FAQs</button></a></div>", unsafe_allow_html=True)

            # if st.button('Try another way'): # and not st.session_state.use_gpt:  # If using HuggingFace and want to offer GPT alternative
            #     st.session_state.button_clicked = True
            #     print('button pressed')
            #     #st.session_state.use_gpt = True  # Switch to GPT for next response
            #     print('changing to gpt')
            #     rag.use_gpt = True
            #     rag.verbose = True
            #     print('sending prompt of ', prompt)
            #     response_text, _ = rag.generate_response(prompt)
            #     print('getting response of', response_text)
                
            #     st.session_state.conversation_history.append({"role": "assistant", "content": response_text, "avatar": avatar_assistant})
            #     bot_message(response_text)
                #st.session_state.use_gpt = False
                #st.experimental_rerun()
           
if __name__ == "__main__":
    run_UI()