import os
import json
import logging
from sentence_transformers import SentenceTransformer
import openai
import replicate
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define API keys
pinecone_api_key = '8d3ee3b5-cfa1-4f73-82c5-45c4833cf93e'
openai.api_key = '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'
replicate_api_token = 'r8_6I9WFvW2YkCDLJhpcGAePJE0Hm36CFk0xR9th'

# Set API keys for services
os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'website-content-index'
namespace = 'ns1'
index = None

def initialize_pinecone_index():
    global index
    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist. Please check the Pinecone index name and try again.")
    else:
        index = pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

def normalize_embedding(embedding):
    min_val = min(embedding)
    max_val = max(embedding)
    range_val = max_val - min_val
    normalized_embedding = [(2 * (x - min_val) / range_val) - 1 for x in embedding]
    return normalized_embedding

def query_vector_db(query):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    logger.info(f"Query embedding before normalization: {query_embedding}")

    # Normalize the embedding to be in the range -1 to 1
    query_embedding = normalize_embedding(query_embedding)
    logger.info(f"Query embedding after normalization: {query_embedding}")

    # Validate the vector
    if not all(isinstance(x, (int, float)) for x in query_embedding):
        raise ValueError("The query embedding contains non-numeric values.")
    
    if not all(-1 <= x <= 1 for x in query_embedding):
        raise ValueError("The query embedding contains values outside the range -1 to 1.")
    
    # Perform search
    try:
        search_result = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=1,
            include_values=True,
            include_metadata=True
        )
        logger.info(f"Search result: {search_result}")
    except Exception as e:
        logger.error(f"Failed to query Pinecone index: {e}")
        raise ValueError(f"Failed to query Pinecone index: {e}")
    
    # Determine the absolute path to the JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, '..', 'scrapers', 'website_content.json')

    # Load scraped data from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Retrieve the relevant content based on the search result
    match_id = int(search_result['matches'][0]['id'])
    urls = list(data.keys())
    search_result_content = data[urls[match_id]]
    
    return search_result_content

def query_llms(prompt, search_results):
    prompt_with_results = f"{prompt}\n\nSearch results:\n{search_results}"

    responses = {}

    try:
        response_gpt35 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_with_results}],
            max_tokens=100
        )
        responses['gpt-3.5-turbo'] = response_gpt35['choices'][0]['message']['content'].strip()
    except Exception as e:
        responses['gpt-3.5-turbo'] = f"Error querying GPT-3.5-turbo: {e}"

    try:
        response_gpt4 = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_with_results}],
            max_tokens=100
        )
        responses['gpt-4'] = response_gpt4['choices'][0]['message']['content'].strip()
    except Exception as e:
        responses['gpt-4'] = f"Error querying GPT-4: {e}"

    try:
        input_llama = {
            "top_p": 1,
            "prompt": prompt_with_results,
            "temperature": 0.5,
            "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            "max_new_tokens": 100
        }
        response_llama = replicate.run("meta/llama-2-70b-chat", input=input_llama)
        responses['llama-2-70b-chat'] = "".join(response_llama)
    except Exception as e:
        responses['llama-2-70b-chat'] = f"Error querying Llama-2-70b-chat: {e}"

    try:
        input_falcon = {
            "prompt": prompt_with_results,
            "temperature": 1
        }
        response_falcon = replicate.run(
            "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
            input=input_falcon
        )
        responses['falcon-40b-instruct'] = "".join(response_falcon)
    except Exception as e:
        responses['falcon-40b-instruct'] = f"Error querying Falcon-40b-instruct: {e}"

    # Evaluate and select the best response
    best_response = evaluate_responses(prompt, responses)

    return best_response, responses

def evaluate_responses(query, responses):
    # Evaluation criteria: Length of Response, Relevance Score, Coherence Score

    def length_of_response(response):
        return len(response.split())

    def relevance_score(response, query_embedding):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        response_embedding = model.encode([response])[0]
        return cosine_similarity([query_embedding], [response_embedding])[0][0]

    def coherence_score(response):
        return response.count('.')  # Simplistic coherence measure based on the number of sentences

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]

    scores = {}
    for model_name, response in responses.items():
        length = length_of_response(response)
        relevance = relevance_score(response, query_embedding)
        coherence = coherence_score(response)
        scores[model_name] = length + relevance + coherence  # Simple aggregate score

    best_model = max(scores, key=scores.get)
    return responses[best_model]