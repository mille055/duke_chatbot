import streamlit as st
from rag import RAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG with environment variables or directly with your keys
rag = RAG()

# # Set Streamlit page configuration to match the desired style
# st.set_page_config(page_title="Duke AIPI Chatbot", layout="wide")

# Main UI function
# def run_UI():
#     # Page styling to match the example (as closely as possible with Streamlit)
#     st.markdown("""
#         <style>
#         .stApp { background-color: #fafafa; }
#         </style>
#     """, unsafe_allow_html=True)

#     st.header("Duke AIPI Chatbot")
#     st.write("What questions do you have about the program?")

#     # Initialize session states for conversation history
#     if 'conversation_history' not in st.session_state:
#         st.session_state['conversation_history'] = []
#     response = ""
#     source = ""

#     # Input field for new query
#     new_query = st.text_input("Ask a question", key='new_query')

#     # Submit button for new query
#     if st.button("Submit") and new_query:
#         # Generate response using RAG
#         response, source = rag.generate_response(new_query)

#         # Update conversation history
#         st.session_state['conversation_history'].append({'query': new_query, 'response': response})

#         # Clear the input field for next query
#         st.session_state['new_query'] = ''
#         st.experimental_rerun()

#     # Display the conversation history
#     for i, convo in enumerate(st.session_state['conversation_history']):
#         st.text_area(f"Question {i+1}", value=convo['query'], height=75, disabled=True)
#         st.text_area(f"Response {i+1}", value=convo['response'], height=150, disabled=True)
#         st.markdown("---")  # Horizontal line for separation

# if __name__ == "__main__":
#     run_UI()

# Streamlit page configuration
st.set_page_config(page_title="Duke AIPI Chatbot", layout="wide")

def run_UI():
    # Apply custom CSS for styling message boxes
    st.markdown("""
        <style>
        .stApp { background-color: #fafafa; }
        .stheader { background-color: #012169}
        .query { 
            background-color: #0577B1; 
            padding: 10px; 
            border-radius: 15px; 
            margin: 10px; 
            float: left;
            clear: both;
        }
        .response { 
            background-color: #ede7f6; 
            padding: 10px; 
            border-radius: 15px; 
            margin: 10px;
            float: left;
            clear: both;
        }
        </style>
    """, unsafe_allow_html=True)

    
    st.header("Duke AIPI Chatbot")
    st.image('assets/duke_picture1.png', caption='Duke University')
    # Initialize or retrieve the conversation history from the session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Text input for the new query
    user_input = st.text_input("Type your message:", key="new_query")

    # Process query on pressing Enter or clicking the Submit button
    if st.button("Submit") or user_input:
        if user_input:  # Ensure non-empty query
            response_text, sources = rag.generate_response(user_input)
            # Append user query and response to conversation history
            st.session_state.conversation_history.append({"type": "query", "text": user_input})
            st.session_state.conversation_history.append({"type": "response", "text": response_text, "sources": sources})
           
    # # Display conversation history
    # for message in st.session_state.conversation_history:
    #     if message["type"] == "query":
    #         st.markdown(f"<div style='text-align: left; background-color:#012169; padding: 10px; border-radius: 15px; margin: 10px;'>{message['text']}</div>", unsafe_allow_html=True)
    #     elif message["type"] == "response":
    #         st.markdown(f"<div style='text-align: left; background-color: #012169; padding: 10px; border-radius: 15px; margin: 10px;'>{message['text']}</div>", unsafe_allow_html=True)
    #         # If there are sources, display a "Learn More" button linking to the first source
    #         if message["sources"]:
    #             st.markdown(f"<div style='text-align: right;'><a href='{message['sources'][0]}' target='_blank'><button style='background-color: #4CAF50; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>Learn More</button></a></div>", unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: left; background-color: #012169; padding: 10px; border-radius: 15px; margin: 10px;'>{response_text}</div>", unsafe_allow_html=True)
            if sources:
                st.markdown(f"<div style='text-align: right;'><a href='{sources[0]}' target='_blank'><button style='background-color: #4CAF50; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>Learn More</button></a></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_UI()

    # # Set up Streamlit page configuration
    # st.set_page_config(page_title="Duke AIPI Chatbot", layout="wide")

    # # Initialize chat messages in session state if not already present
    # if "messages" not in st.session_state:
    #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # # Function to clear chat history
    # def clear_chat_history():
    #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # # Sidebar button to clear chat history
    # st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # # Display chat messages
    # for message in st.session_state.messages:
    #     with st.chat_message_container():
    #         if message["role"] == "user":
    #             st.caption("You")
    #         else:  # "assistant"
    #             st.caption("Assistant")
    #         st.write(message["content"])

    # # User input through chat_input
    # user_input = st.chat_input("Ask me something...", key="chat_input")

    # # Process user input
    # if user_input:
    #     # Append user's message to chat history
    #     st.session_state.messages.append({"role": "user", "content": user_input})
        
    #     # Generate response from RAG
    #     response_text, sources = rag.generate_response(user_input)
        
    #     # Append assistant's response to chat history
    #     st.session_state.messages.append({"role": "assistant", "content": response_text})
        
    #     # Optionally, handle "Learn More" action for the first source link
    #     if sources:
    #         learn_more_url = sources[0]
    #         st.session_state.messages.append({"role": "assistant", "content": f"[Learn More]({learn_more_url})"})

    # if __name__ == "__main__":
    #     run_UI()
