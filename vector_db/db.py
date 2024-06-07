import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define API keys and other constants
pinecone_api_key = '8d3ee3b5-cfa1-4f73-82c5-45c4833cf93e'
index_name = 'website-content-index'
namespace = 'ns1'

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

def store_data():
    try:
        # Verify index existence
        if index_name not in pc.list_indexes().names():
            logger.error(f"Index '{index_name}' does not exist. Please check the Pinecone index name and try again.")
            return
        
        # Connect to the existing index
        index = pc.Index(index_name)
        logger.info(f"Connected to existing index '{index_name}'")

        # Determine the absolute path to the JSON file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, '..', 'scrapers', 'website_content.json')

        # Load scraped data from JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Log data keys to ensure JSON is loaded correctly
        logger.info(f"Loaded data keys: {list(data.keys())}")

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Prepare data for vector storage
        texts = [content for url, content in data.items()]
        logger.info(f"Prepared {len(texts)} texts for embedding")

        # Create embeddings
        embeddings = model.encode(texts)
        logger.info(f"Created embeddings for {len(embeddings)} texts")

        # Prepare vectors for upsert
        vectors = [{"id": str(i), "values": embeddings[i].tolist()} for i in range(len(texts))]

        # Upsert vectors to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)
        logger.info("Data storage process completed.")

    except Exception as e:
        # Log any errors that occur during data storage
        logger.error(f"An error occurred during data storage: {e}")

if __name__ == "__main__":
    store_data()
