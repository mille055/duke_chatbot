# Duke Chat
## aipi 590 duke chatbot for project 2


This project includes chat bot using a fine-tuned local LLM and a *Retrieval Augmented Generation* (RAG) System to chat with documents from the Duke University AI Master's program. It creates an interactive chat experience with the data gleaned from the Duke websites. Inspiration drawn from https://github.com/architkaila/Chat-With-Documents and https://github.com/guptashrey/Duke-ChatBot.

![chatbot image](./assets/duke_picture1.png)

## Overview

1. **Chat agent**: Users can ask any questions regarding the Duke AIPI program, including application process, course overview, faculty members and career resources.
2. **Text Preparation**: We scraped the textual information from the AIPI program websites as well as the FAQ documentation. We updated the FAQ for the upcoming academic year (Fall 2024). 
4. **Embedding and Indexing**:  We used OpenAI embedding model "text-embedding-3-small" to embed the textual information into vector form. We then store them into Pinecone vector database. When the user query in a question, our pipeline uses the same embedding model to convert user query into vector form and then pass in similarity search to find the top 3 similar textual inforamtion from the data. We chose Pinecone for its speed, scalability, and low cost. 
5. **LLM**: We fine-tuned [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1) for question and answering functionality by using a list of query and answer pair from the data sources. We chose Mistral-7b because it is known for its high performance despite its small size, besting larger LLama-2 models on a range of tasks. It is also known for being readily fine-tuned, as evidenced by the Mistral-7B Instruct model. The model is hosted as an endpoint at Hugging Face.  \nNote that the FAQ text is stored in a separate index, and a semantic search is first performed over the FAQs for each user query; if there is a match, the text answer from the FAQs is displayed with a link to the FAQ page hosted on this repository (https://mille055.github.io/duke_chatbot/data/faqs.html)--this helps the user experience to receive prompt, accurate responses if the query can be addressed by the FAQs. Otherwise, the LLM model is used to generate a resopnse.
6. **Streamlit Interface**: Duke themed chatbot that generates response with fine-tuned model and user query. The endpoint for the LLM is currently set for continuous operation. During development, there was a spin down process for inactive periods, with some time for start up when the model was needed again. The user interface would display a dedicated message if the model was unavailable, with options of waiting or chaning a toggle to temporarily use chatgpt rather than our custom fine-tuned model.
7. **Video Descripotion**: There is a youtube video which describes our approach and covers a brief cost analysis, which can be located at https://www.youtube.com/watch?v=GrMB4Fetx7o. A couple comments about the cost analysis: 1. several of the estimates, such as for pinecone expenses, may be overestimates depending on the amount of usage. In setting up and uisng the vector database, our actual expenses have been about $0.01/day for just the limited use. 2. The openAI expenses include the monthly account fee, which could be spread over different projects. 3. We could save on our Hugging Face endpoint expenses (the highest cost item) by operating with a spin down, but for the interest of usability we have opted for continuous availabilty of the model.

&nbsp;
## Running the Application 
The streamlit application is deployed at https://duke-aipi-chatbot.azurewebsites.net. However, if you are using this repository locally and are incorporating elements for your own application there are a few steps:

Following are the steps to run the StreamLit Application: 

**1. Create a new conda environment and activate it:** 
```
conda create --name chat python=3.9.10
conda activate chat
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**4. Add OpenAI API Key and other keys**
```
- Copy the `env.example` file and rename the copy to `.env`.
- Open the `.env` file and replace `your_api_key_here` with your actual OpenAI API key 

Add your own Pinecone and HuggingFace keys and tokens as well if you are using your resources.  
```
**5. Run the application**
```
streamlit run app.py
```


## Directory Structure
```
.
├── README.md
├── app.py
├── assets
│   ├── AIPI-Incoming-Student-FAQ.docx
│   ├── Blue_question_mark_icon.png
│   ├── duke_chapel_blue.png
│   ├── duke_chapel_blue_with_text.png
│   ├── duke_d.png
│   ├── duke_d_2.png
│   ├── duke_logo_pic.png
│   ├── duke_picture1.jpg
│   ├── duke_picture1.png
│   ├── duke_picture2.png
│   ├── duke_picture3.png
│   ├── empty.txt
│   └── mashup_duke_image_title.png
├── data
│   ├── AIPI-Incoming-Student-FAQ.docx
│   ├── AIPI-Incoming-Student-FAQ.txt
│   ├── archive
│   │   ├── app_old.py
│   │   └── extracted_data_2024-03-31_12-20-18.json
│   ├── extracted_data_2024-04-01_07-59-36.json
│   ├── extracted_data_from_faq.json
│   ├── faqs.html
│   ├── fine_tune_data.json
│   ├── q_a_list.jsonl
│   └── updated_qa_pairs.json
├── example.env
├── extract_text.py
├── extract_text_from_faqdoc.py
├── faq_to_html.py
├── faq_to_pinecone.py
├── notebooks
│   ├── Copy_of_Create_ft_data.ipynb
│   ├── Copy_of_chatbot_Finetune_notebook0412.ipynb
│   ├── MergeModels.ipynb
│   ├── Mistral_7B_finetune_0413.ipynb
│   ├── chatbot_Finetune_notebook.ipynb
│   └── chatbot_Finetune_notebook2.ipynb
├── rag.py
├── requirements.txt
└── tree.txt

5 directories, 39 files
```