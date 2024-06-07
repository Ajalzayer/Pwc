import requests
from bs4 import BeautifulSoup
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = '\n'.join([p.get_text() for p in paragraphs])
        return text_content
    except requests.RequestException as e:
        logger.error(f"Failed to scrape website {url}: {e}")
        return None

def main():
    urls = [
        "https://u.ae/en/information-and-services",
        "https://u.ae/en/information-and-services/visa-and-emirates-id",
        "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas",
        "https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas/golden-visa"
    ]

    data = {}
    for url in urls:
        logger.info(f"Scraping: {url}")
        scraped_data = scrape_website(url)
        if scraped_data is not None:
            data[url] = scraped_data

    # Save the scraped data to a JSON file
    output_file = 'website_content.json'
    with open(output_file, 'w') as f:
        json.dump(data, f)
    logger.info(f"Scraped data saved to {output_file}")

if __name__ == "__main__":
    main()