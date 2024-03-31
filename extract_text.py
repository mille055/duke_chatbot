import logging
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_url_content(url):
    """
    Fetches content from a URL with improved error handling.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.text
    except requests.HTTPError as e:
        logging.error(f"HTTP error occurred for {url}: {e}")
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for {url}: {e}")
    return None

def extract_text_from_html(html_content):
    """
    Extracts and cleans text from HTML content with additional cleaning steps.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    
    # Optional: Additional cleaning steps
    # Example: Replace multiple spaces with a single space
    text = ' '.join(text.split())
    
    return text

def fetch_and_process_url(url):
    """
    Orchestrates fetching the HTML content and extracting clean text from it.
    """
    logging.info(f"Processing {url}")
    html_content = fetch_url_content(url)
    text = extract_text_from_html(html_content)
    return text

# def process_urls_concurrently(urls, max_workers=5):
#     """
#     Uses threading to process multiple URLs concurrently.
#     """
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_url = {executor.submit(fetch_and_process_url, url): url for url in urls}
#         for future in future_to_url:
#             url = future_to_url[future]
#             try:
#                 data = future.result()
#                 # Optional: Save or process the data here
#             except Exception as exc:
#                 logging.error(f'{url} generated an exception: {exc}')
#             else:
#                 logging.info(f'{url} page length: {len(data)}')


def save_to_json(data, filename="extracted_data.json"):
    """
    Saves extracted text data to a JSON file.
    
    Args:
        data (dict): A dictionary where each key is a URL or filename and the value is the extracted text.
        filename (str): Name of the file to save the data.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully saved to {filename}")

def process_urls_concurrently(urls, max_workers=5):
    """
    Processes multiple URLs concurrently and saves the results to a JSON file.
    """
    extracted_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_and_process_url, url): url for url in urls}
        for future in future_to_url:
            url = future_to_url[future]
            try:
                text = future.result()
                extracted_data[url] = text
            except Exception as exc:
                logging.error(f'{url} generated an exception: {exc}')
            else:
                logging.info(f'{url} processed successfully')

    # Save the extracted data to JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_to_json(extracted_data, filename=f"extracted_data_{timestamp}.json")


# Example usage
if __name__ == "__main__":
    urls = [
        "https://ai.meng.duke.edu/degree"
        # Room for more URLs as needed
    ]
    process_urls_concurrently(urls)
