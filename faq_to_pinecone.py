import json
import re
from pinecone import init, Index, Pinecone, ServerlessSpec
from dotenv import load_dotenv
from rag import RAG  


# function to generate embeddings
def generate_embeddings(rg, text):
    print('getting embeddings for', text)
    response = rg.openai_client.embeddings.create(
            model=rg.openai_embedding_model,
            input=[text]
            )
    embedding = response.data[0].embedding
    return embedding


if __name__ == "__main__":
    
    # instantiate RAG class with pinecone index name specified
    rag = RAG(pinecone_index_name = 'faq-database')

    # read the FAQ document
    filename = "data/AIPI-Incoming-Student-FAQ.txt"
    with open(filename, "r") as file:
        faq_text = file.read()

   #print(f'faq_text is {faq_text}')
    
    # Split the document into Q&A pairs
    qa_pairs = re.split(r"\n\d+\.\s*", faq_text)[1:]
    #print(qa_pairs)

    # create the json object
    faq_json_objects = []
    for idx, qa_pair in enumerate(qa_pairs):
        print('qa_pair is', qa_pair)
        # Split the QA pair into question and answer
        qa_parts = re.split(r"\?", qa_pair.strip(), maxsplit=1)
        print('qa_parts are', qa_parts, len(qa_parts))
        # Check if there are at least two parts (question and answer)
        if len(qa_parts) >= 2:
            question, answer = qa_parts
            
            # Generate embeddings for question and answer
            question_embedding = generate_embeddings(rag, question)
            answer_embedding = generate_embeddings(rag, answer)
            qa_text = str(question + ' ' + answer)
            qa_embedding = generate_embeddings(rag, qa_text)
            
            print('question_embedding is of type', type(question_embedding))
            print('answer embedding is of type', type(answer_embedding))
            print('length of answer embedding is', len(answer_embedding))

            # Create JSON object with question, answer, and embeddings
            faq_json_objects.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "question_embedding": question_embedding,
                "answer_embedding": answer_embedding,
                "qa_embedding": qa_embedding
            })
            print('created entry for question', question.strip())

            # Upsert each JSON object into the Pinecone database
            #rag.index.upsert(f"faq_{idx}", json.dumps(faq_json_objects[-1]))
            rag.index.upsert(vectors=[{"id": str(idx), "values": qa_embedding,  "metadata":{"question": question, "answer": answer}}])
        else:
            # Handle cases where the QA pair doesn't have enough parts
            print(f"Skipping invalid QA pair at index {idx}: {qa_pair}")

    print("FAQ data upserted into Pinecone database successfully.")