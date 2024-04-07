import json
import re

def process_faq_text(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as file:
        faq_text = file.read()

    # Regex pattern to identify questions and answers
    pattern = re.compile(r'(\d+\.\s+)(.*?)\n(.*?)(?=\n\d+\.|\Z)', re.DOTALL)

    faqs = []
    matches = pattern.findall(faq_text)
    for match in matches:
        question_number, question, answer = match
        faqs.append({
            "question": question.strip(),
            "answer": answer.strip().replace('\n', ' ')
        })

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump({"FAQs": faqs}, json_file, indent=4, ensure_ascii=False)

if __name__=='__main__':

    process_faq_text('data/AIPI-Incoming-Student-FAQ.txt', 'extracted_data_from_faq.json')
