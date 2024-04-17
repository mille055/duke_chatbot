# Duke Chat
## aipi 590 duke chatbot for project 2


This project includes chat bot using a fine-tuned local LLM and a *Retrieval Augmented Generation* (RAG) System to chat with documents from the Duke University AI Master's program. It creates an interactive chat experience with the data gleaned from the Duke websites. Inspiration drawn from https://github.com/architkaila/Chat-With-Documents.

![chatbot image](./assets/duke_picture1.png)

## Overview

1. **Chat agent**: Users can ask any questions regarding the Duke AIPI program, including application process, course overview, faculty members and career resources.
2. **Text Preparation**: We scraped the textual information from the AIPI program website as well as the FAQ documentation
4. **Embedding and Indexing**:  We used OpenAI embedding model "text-embedding-3-small" to embed the textual information into vector form. We then store them into Pinecone vector database. When the user query in a question, our pipeline uses the same embedding model to convert user query into vector form and then pass in similarity search to find the top 3 similar textual inforamtion from the data.
5. **LLM**: We fine-tuned [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1) for question and answering functionality by using a list of query and answer pair from the data sources
6. **Streamlit Interface**: Duke themed chatbot that generates response with fine-tuned model and user query

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

