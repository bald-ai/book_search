from flask import Flask, render_template, request, jsonify
import os
import logging

# Import the search function from search_chunks.py
from search_chunks import search_book_chunks

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the directory where embedding files are stored
EMBEDDINGS_DIR = "embedded_books"

@app.route('/')
def index():
    """Renders the main HTML page."""
    # We could pass the list of available books here if needed,
    # but the current JS handles it based on expected filenames.
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles search requests from the frontend."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')
    book_filename = data.get('book_filename')

    if not query or not book_filename:
        return jsonify({"error": "Missing 'query' or 'book_filename' in request"}), 400

    # Construct the full path to the embeddings file
    # Basic security check: ensure filename doesn't contain path traversal chars
    if '..' in book_filename or '/' in book_filename:
         logging.warning(f"Attempted path traversal with filename: {book_filename}")
         return jsonify({"error": "Invalid book filename"}), 400

    embeddings_file_path = os.path.join(EMBEDDINGS_DIR, book_filename)

    # Check if the file actually exists
    if not os.path.isfile(embeddings_file_path):
        logging.error(f"Embeddings file not found: {embeddings_file_path}")
        return jsonify({"error": "Selected book data not found on server."}), 404

    logging.info(f"Searching in '{embeddings_file_path}' for query: '{query}'")

    try:
        results = search_book_chunks(query, embeddings_file_path)
        logging.info(f"Found {len(results)} results for query: '{query}' in '{book_filename}'")
        
        # Select only the top result if results are found
        top_result = results[0] if results else None

        # Format results slightly for frontend if needed, or return directly
        # Ensure scores are JSON serializable (already handled in search_book_chunks)
        formatted_results = []
        if top_result:
            formatted_results.append(
                {
                    "text": top_result.get("text", ""),
                    "chunk_index": top_result.get("chunk_index", -1),
                    "score": top_result.get("score", 0.0),
                    # Add other fields like percent_into_book if needed
                    "percent_into_book": top_result.get("percent_into_book")
                }
            )
            logging.info(f"Returning top result (Chunk {top_result.get('chunk_index')}) for query: '{query}' in '{book_filename}'")
        else:
            logging.info(f"No results found to return for query: '{query}' in '{book_filename}'")

        return jsonify(formatted_results) # Return list containing zero or one result
    except Exception as e:
        logging.error(f"Error during search for query '{query}' in '{book_filename}': {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during search."}), 500


if __name__ == '__main__':
    # Ensure the embeddings directory exists
    if not os.path.isdir(EMBEDDINGS_DIR):
        logging.error(f"Embeddings directory '{EMBEDDINGS_DIR}' not found. Please create it and add embedding files.")
    else:
        logging.info(f"Embeddings directory found at: {os.path.abspath(EMBEDDINGS_DIR)}")
    app.run(debug=True) 