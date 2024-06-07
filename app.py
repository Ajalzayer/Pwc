from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from llm_query.query import initialize_pinecone_index, query_vector_db, query_llms
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('query_llms')
def handle_query_llms(data):
    prompt = data['prompt']
    try:
        initialize_pinecone_index()  # Ensure the index is initialized
        search_results = query_vector_db(prompt)
        best_response, all_responses = query_llms(prompt, search_results)
        emit('response', {'best_response': best_response, 'all_responses': all_responses})
    except ValueError as e:
        logger.error(e)
        emit('error', {'error': str(e)})

if __name__ == "__main__":
    socketio.run(app, debug=True)