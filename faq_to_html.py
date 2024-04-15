import json, re, os

def json_to_html(faq_json, output_html):
    # Load JSON data
    with open(faq_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Start HTML document
    html_content = """
    <html>
    <head>
        <title>FAQs</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .faq { margin-bottom: 20px; }
            .question { color: #00539B; font-weight: bold; }
            .answer { color: #333; }
            /* Add your CSS styles here */
            .stApp { background-color: #fafafa; }
            .stheader { background-color: #0577B1}
            .user-message, .bot-message {
                background-color: #00539B; 
                padding: 20px; 
                border-radius: 15px; 
                margin: 15px; 
                color: white;
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
        </style>
    </head>
    <body>
        <img src='/assets/duke_chapel_blue.png' alt='Header Image' style='width:100%; height:auto;'>
        <h1>Frequently Asked Questions (FAQs)</h1>
    """
    
    # Add each FAQ
    for faq in data["FAQs"]:
        html_content += f"""
        <div class='faq'>
            <div class='question'>{faq['question']}</div>
            <div class='answer'>{faq['answer']}</div>
        </div>
        """
    
    # Close HTML document
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as file:
        file.write(html_content)

if __name__ == "__main__":
    json_to_html('data/extracted_data_from_faq.json', 'faqs.html')
