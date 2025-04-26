from flask import Flask, render_template, request, jsonify
import os
import logging
import json
import datetime # Import datetime for timestamping

# Import the search function from search_chunks.py
from search_chunks import search_book_chunks

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the directory where embedding files are stored
EMBEDDINGS_DIR = "embedded_books"
FEEDBACK_FILE = 'feedback_data.json'
ALL_QUERIES_FILE = 'all_queries.json' # Define the new log file name

# Helper function to load feedback data
def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading feedback file: {e}")
        return [] # Return empty list on error

# Helper function to save feedback data
def save_feedback(data):
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving feedback file: {e}")

# Helper function to load query log data
def load_query_log():
    if not os.path.exists(ALL_QUERIES_FILE):
        return []
    try:
        with open(ALL_QUERIES_FILE, 'r') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading query log file: {e}")
        return [] # Return empty list on error

# Helper function to save query log data
def save_query_log(data):
    try:
        with open(ALL_QUERIES_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving query log file: {e}")

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Get list of available book embedding files
    available_books = []
    if os.path.exists(EMBEDDINGS_DIR):
        for filename in os.listdir(EMBEDDINGS_DIR):
            if filename.endswith('_embeded.json'):
                # Extract a 'cleaner' book name for display if possible
                book_name = filename.replace('_embeded.json', '').replace('_', ' ').title()
                available_books.append({'filename': filename, 'display_name': book_name})
    
    return render_template('index.html', available_books=available_books)

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

    # --- Log the query ---
    # Prepare for multi-book logging, even if only one is sent currently
    selected_books_list = [book_filename] # Create a list from the single filename

    try:
        all_queries = load_query_log()
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "selected_books": selected_books_list, # Log as a list
            "query": query
        }
        all_queries.append(log_entry)
        save_query_log(all_queries)
        logging.info(f"Logged query for books: {selected_books_list}") # Log the list
    except Exception as e:
        logging.error(f"Failed to log query: {e}", exc_info=True)
    # --- End logging ---

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

    # Signal search start (for backend logs, frontend will log on receiving response)
    logging.info(f"Initiating search_book_chunks for query: '{query}'")

    try:
        results = search_book_chunks(query, embeddings_file_path)
        logging.info(f"search_book_chunks completed. Found {len(results)} potential results.")

        # Select only the top result if results are found
        top_result = results[0] if results else None

        # Format results slightly for frontend if needed, or return directly
        formatted_results = []
        search_completed_flag = True # Flag indicating the search function ran fully
        total_chunks = 0 # Initialize total_chunks
        chunk_text = None
        if top_result:
            # Always get the chunk text from chunks_json
            original_json = book_filename.replace('_embeded.json', '.json')
            original_path = os.path.join('chunks_json', original_json)
            chunk_index = top_result.get('chunk_index', -1)
            if os.path.isfile(original_path) and chunk_index != -1:
                try:
                    with open(original_path, 'r', encoding='utf-8') as f:
                        original_chunks_data = json.load(f) # Load the whole file
                    total_chunks = len(original_chunks_data) # Get total chunk count
                    # Find the specific chunk's text
                    for chunk in original_chunks_data:
                        if chunk.get('chunk_index') == chunk_index:
                            chunk_text = chunk.get('text', None)
                            break
                except Exception as e:
                    logging.error(f"Error loading chunk text or counting chunks from {original_path}: {e}")

            if chunk_text is not None: # Check if chunk_text was found
                formatted_results.append(
                    {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "score": top_result.get("score", 0.0),
                        "rank": top_result.get("rank")
                    }
                )
                logging.info(f"Returning top result (Chunk {chunk_index}/{total_chunks}, Rank: {top_result.get('rank')}) for query: '{query}' in '{book_filename}'")
            else:
                 logging.warning(f"Could not find text for chunk index {chunk_index} in {original_path}, although search returned it.")
                 # Still return search_completed and total_chunks if applicable, but results might be empty
                 # Do not append to formatted_results if text is None
        else:
            logging.info(f"No results found to return for query: '{query}' in '{book_filename}'")
            # If no results, we might still want to know the total chunks if the file existed
            # Let's try to load the original file just for the count if it wasn't loaded before
            original_json = book_filename.replace('_embeded.json', '.json')
            original_path = os.path.join('chunks_json', original_json)
            total_chunks = 0
            if os.path.isfile(original_path):
                 try:
                    with open(original_path, 'r', encoding='utf-8') as f:
                        original_chunks_data = json.load(f)
                    total_chunks = len(original_chunks_data)
                 except Exception as e:
                     logging.error(f"Error reading {original_path} just for chunk count: {e}")

        # Include the completion flag and total chunks in the response
        return jsonify({
            "results": formatted_results, # Will be empty list if no chunk found
            "search_completed": search_completed_flag,
            "total_chunks": total_chunks # Return total chunk count
        })
    except Exception as e:
        logging.error(f"Error during search for query '{query}' in '{book_filename}': {e}", exc_info=True)
        # Indicate search did not complete successfully in the response
        return jsonify({"error": "An internal error occurred during search.", "search_completed": False}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    query = data.get('query')
    chunk_index = data.get('chunk_index')
    is_correct = data.get('is_correct')
    # Expect a list of book filenames now
    selected_books = data.get('selected_books') # Changed from book_filename
    rank = data.get('rank')

    # Validate the new structure
    if query is None or chunk_index is None or is_correct is None or selected_books is None or not (
        rank is None or (isinstance(rank, int) and rank >= 0)
    ):
        logging.warning(f"Received incomplete or invalid feedback data: {data}")
        # Updated error message
        return jsonify({"error": "Missing or invalid required feedback fields (query, chunk_index, is_correct, selected_books, rank[must be null or >= 0])"}), 400

    # Ensure selected_books is a list
    if not isinstance(selected_books, list) or not all(isinstance(item, str) for item in selected_books):
         logging.warning(f"Invalid format for selected_books: {selected_books}")
         return jsonify({"error": "'selected_books' must be a list of strings"}), 400

    feedback_entry = {
        "query": query,
        "selected_books": selected_books, # Store the list
        "chunk_index": chunk_index,
        "is_correct": is_correct,
        "rank": rank
        # Optional: Add timestamp
        # "timestamp": datetime.datetime.now().isoformat()
    }

    logging.info(f"Received feedback: {feedback_entry}")

    # Load existing feedback, append new entry, and save
    all_feedback = load_feedback()
    all_feedback.append(feedback_entry)
    save_feedback(all_feedback)

    return jsonify({"message": "Feedback received successfully"}), 200

# --- New Endpoint to Get Specific Chunk Text ---
@app.route('/get_chunk_text', methods=['POST'])
def get_chunk_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    book_filename_embed = data.get('book_filename') # Expecting the _embeded.json name
    chunk_index = data.get('chunk_index')

    if book_filename_embed is None or chunk_index is None:
        return jsonify({"error": "Missing 'book_filename' or 'chunk_index'"}), 400

    try:
        chunk_index = int(chunk_index)
    except ValueError:
        return jsonify({"error": "'chunk_index' must be an integer"}), 400

    # Derive the original chunk filename
    if not book_filename_embed.endswith('_embeded.json'):
         return jsonify({"error": "Invalid book filename format"}), 400
    original_json_filename = book_filename_embed.replace('_embeded.json', '.json')
    original_path = os.path.join('chunks_json', original_json_filename)

    if '..' in original_json_filename or '/' in original_json_filename:
         logging.warning(f"Attempted path traversal with derived filename: {original_json_filename}")
         return jsonify({"error": "Invalid book filename"}), 400

    if not os.path.isfile(original_path):
        logging.error(f"Original chunk file not found: {original_path}")
        return jsonify({"error": "Book chunk data not found"}), 404

    try:
        with open(original_path, 'r', encoding='utf-8') as f:
            original_chunks_data = json.load(f)

        found_chunk = None
        for chunk in original_chunks_data:
            if chunk.get('chunk_index') == chunk_index:
                found_chunk = chunk
                break

        if found_chunk:
            return jsonify({
                "text": found_chunk.get("text", "[Text not available]"),
                "chunk_index": found_chunk.get("chunk_index"),
            })
        else:
            logging.warning(f"Chunk index {chunk_index} not found in {original_path}")
            return jsonify({"error": f"Chunk index {chunk_index} not found"}), 404

    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from: {original_path}")
        return jsonify({"error": "Error reading book data"}), 500
    except Exception as e:
        logging.error(f"Error processing request for chunk {chunk_index} in {original_path}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    # Ensure the embeddings directory exists
    if not os.path.isdir(EMBEDDINGS_DIR):
        logging.error(f"Embeddings directory '{EMBEDDINGS_DIR}' not found. Please create it and add embedding files.")
    else:
        logging.info(f"Embeddings directory found at: {os.path.abspath(EMBEDDINGS_DIR)}")
    app.run(debug=True) 