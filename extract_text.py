import logging
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

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

def process_urls_concurrently(urls, max_workers=5):
    """
    Uses threading to process multiple URLs concurrently.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_and_process_url, url): url for url in urls}
        for future in future_to_url:
            url = future_to_url[future]
            try:
                data = future.result()
                # Optional: Save or process the data here
            except Exception as exc:
                logging.error(f'{url} generated an exception: {exc}')
            else:
                logging.info(f'{url} page length: {len(data)}')

# Example usage
if __name__ == "__main__":
    urls = [
        "https://example.com",
        "https://www.python.org",
        # Add more URLs as needed
    ]
    process_urls_concurrently(urls)
