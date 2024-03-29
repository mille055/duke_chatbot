# Duke Chat
## aipi 590 duke chatbot for project 2


This project includes chat bot using a fine-tuned local LLM and a *Retrieval Augmented Generation* (RAG) System to chat with documents from the Duke University AI Master's program. It creates an interactive chat experience with the data gleaned from the Duke websites. Inspiration drawn from https://github.com/architkaila/Chat-With-Documents.

![chatbot image](./assets/duke-image.png)

## Overview

1. **Chat agent**: Users can chat with
2. **Text Preparation**: 
4. **Embedding and Indexing**:  
5. **Streawmlit Interface**: 

&nbsp;
## Running the Application 
Following are the steps to run the StreamLit Application: 

**1. Create a new conda environment and activate it:** 
```
conda create --name chat python=3.8.17
conda activate chat
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**4. Add OpenAI API Key**
```
- Copy the `env.example` file and rename the copy to `.env`.
- Open the `.env` file and replace `your_api_key_here` with your actual OpenAI API key.
```
**5. Run the application**
```
streamlit run app.py
```


## Directory Structure

