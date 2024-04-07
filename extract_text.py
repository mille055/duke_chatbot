import logging
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from urllib.parse import urlparse, urljoin
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Extract text from all pages within a domain.')
    parser.add_argument('-u', '--url', default='https://ai.meng.duke.edu/degree', help='The main URL to start crawling from. Defaults to https://ai.meng.duke.edu/degree')
    args = parser.parse_args()
    return args.url


def is_valid_url(url, domain_name):
    """
    Checks if a URL is valid and belongs to the specified domain.
    """
    parsed_url = urlparse(url)
    return bool(parsed_url.netloc) and domain_name in parsed_url.netloc


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

def get_all_website_links(url):
    """
    Returns all URLs that are found on 'url' in which it belongs to the same website
    """
    # Domain name of the URL without the protocol
    domain_name = urlparse(url).netloc
    urls = set()  # All discovered URLs
    visited_urls = set()  # Set of visited URLs to avoid processing a page more than once
    urls.add(url)
    visited_urls.add(url)

    session = requests.Session()
    session.headers["User-Agent"] = "Googlebot/2.1 (+http://www.google.com/bot.html)"
    
    while urls:
        current_url = urls.pop()
        print(f"Crawling: {current_url}")
        try:
            response = session.get(current_url)
            response.raise_for_status()  # ensure we notice bad responses
        except (requests.RequestException, ValueError):
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                # href empty tag
                continue
            href = urljoin(current_url, href)
            parsed_href = urlparse(href)
            # remove URL GET parameters, URL fragments, etc.
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            
            if not is_valid_url(href, domain_name):
                # not a valid URL
                continue
            if href in visited_urls:
                # already visited URL
                continue
            visited_urls.add(href)
            urls.add(href)
    
    return visited_urls



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
    save_to_json(extracted_data, filename=f"data/extracted_data_{timestamp}.json")


# Example usage
if __name__ == "__main__":
    main_url = parse_arguments()
    all_links = get_all_website_links(main_url)

    process_urls_concurrently(all_links)
    