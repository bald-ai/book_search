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
    # Recursively get list of available book embedding files (with subfolders)
    available_books = []
    if os.path.exists(EMBEDDINGS_DIR):
        for root, dirs, files in os.walk(EMBEDDINGS_DIR):
            for filename in files:
                if filename.endswith('_embeded.json'):
                    # Get the path relative to EMBEDDINGS_DIR
                    rel_path = os.path.relpath(os.path.join(root, filename), EMBEDDINGS_DIR)
                    book_name = filename.replace('_embeded.json', '').replace('_', ' ').title()
                    available_books.append({'filename': rel_path, 'display_name': book_name})
    return render_template('index.html', available_books=available_books)

@app.route('/search', methods=['POST'])
async def search():
    """Handles search requests from the frontend."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')
    # Expecting a list of relative paths (filenames) now
    book_filenames = data.get('book_filenames') 

    # Validate input
    if not query or not book_filenames:
        return jsonify({"error": "Missing 'query' or 'book_filenames' in request"}), 400
    if not isinstance(book_filenames, list) or not book_filenames:
         return jsonify({"error": "'book_filenames' must be a non-empty list"}), 400
    if not all(isinstance(fn, str) for fn in book_filenames):
         return jsonify({"error": "'book_filenames' must contain only strings"}), 400

    # --- Log the query ---
    selected_books_list = book_filenames 
    try:
        all_queries = load_query_log()
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "selected_books": selected_books_list, # Log the list
            "query": query
        }
        all_queries.append(log_entry)
        save_query_log(all_queries)
        logging.info(f"Logged query for books: {selected_books_list}") # Log the list
    except Exception as e:
        logging.error(f"Failed to log query: {e}", exc_info=True)
    # --- End logging ---

    # Construct the full paths to the embeddings files and validate them
    embeddings_file_paths = []
    valid_filenames_for_chunk_lookup = {} # Store mapping for later chunk text lookup

    for filename in book_filenames:
        # Security: allow subfolders, but prevent path traversal
        if '..' in filename or filename.startswith('/') or '\\' in filename:
            logging.warning(f"Attempted path traversal with filename: {filename}")
            return jsonify({"error": f"Invalid book filename format: {filename}"}), 400
        if not filename.endswith('_embeded.json'):
            logging.warning(f"Invalid filename extension: {filename}")
            return jsonify({"error": f"Invalid book filename extension: {filename}"}), 400
        full_path = os.path.join(EMBEDDINGS_DIR, filename)
        if not os.path.isfile(full_path):
            logging.error(f"Required embeddings file not found: {full_path}")
            return jsonify({"error": f"Selected book data not found on server: {filename}"}), 404
        embeddings_file_paths.append(full_path)
        # Store a mapping from the base name (assumed to match book_name) to the original relative filename
        base_name = os.path.splitext(os.path.basename(filename))[0].replace('_embeded', '')
        valid_filenames_for_chunk_lookup[base_name] = filename

    logging.info(f"Searching in {len(embeddings_file_paths)} files for query: '{query}'")
    logging.info(f"Initiating search_book_chunks for query: '{query}' across {len(embeddings_file_paths)} books")

    try:
        results = await search_book_chunks(query, embeddings_file_paths)
        logging.info(f"search_book_chunks completed. Found {len(results)} potential results.")
        top_result = results[0] if results else None
        formatted_results = []
        search_completed_flag = True 
        total_chunks = 0
        chunk_text = None
        result_book_name = None
        if top_result:
            chunk_index = top_result.get('chunk_index', -1)
            result_book_name = top_result.get('book_name')
            result_rank = top_result.get('rank')
            result_score = top_result.get('score', 0.0)
            if result_book_name and chunk_index != -1:
                lookup_base_name = result_book_name.lower().replace(' ', '_')
                # Find the filename (with subfolder) that matches this book
                original_filename_embed = None
                for base, rel_path in valid_filenames_for_chunk_lookup.items():
                    if base == lookup_base_name:
                        original_filename_embed = rel_path
                        break
                if original_filename_embed:
                    # Replace only the filename, not the folder, to get the chunk file path
                    chunk_json_rel = original_filename_embed.replace('_embeded.json', '.json')
                    original_path = os.path.join('chunks_json', chunk_json_rel)
                    if os.path.isfile(original_path):
                        try:
                            with open(original_path, 'r', encoding='utf-8') as f:
                                original_chunks_data = json.load(f)
                            total_chunks = len(original_chunks_data)
                            for chunk in original_chunks_data:
                                if chunk.get('chunk_index') == chunk_index:
                                    chunk_text = chunk.get('text', None)
                                    break
                        except Exception as e:
                            logging.error(f"Error loading chunk text or counting chunks from {original_path}: {e}")
                    else:
                        logging.warning(f"Original chunk file not found: {original_path} derived from book_name '{result_book_name}'")
                else:
                    logging.warning(f"Could not map returned book_name '{result_book_name}' (lookup key: '{lookup_base_name}') back to a valid input filename.")
            if chunk_text is not None:
                formatted_results.append(
                    {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "score": result_score,
                        "rank": result_rank,
                        "book_name": result_book_name,
                        "total_chunks": total_chunks,
                        "book_filename": original_filename_embed
                    }
                )
                logging.info(f"Returning top result (Book: '{result_book_name}', Chunk {chunk_index}/{total_chunks}, Rank: {result_rank}) for query: '{query}'")
            else:
                logging.warning(f"Could not find text for chunk index {chunk_index} in book '{result_book_name}', although search returned it.")
                total_chunks = 0
        else:
            logging.info(f"No results found to return for query: '{query}' across selected books.")
            total_chunks = 0
        return jsonify({
            "results": formatted_results,
            "search_completed": search_completed_flag,
            "total_chunks": total_chunks if not formatted_results else formatted_results[0].get("total_chunks", 0)
        })
    except Exception as e:
        logging.error(f"Error during search for query '{query}': {e}", exc_info=True)
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
    book_filename_embed = data.get('book_filename') 
    chunk_index = data.get('chunk_index')
    if book_filename_embed is None or chunk_index is None:
        return jsonify({"error": "Missing 'book_filename' or 'chunk_index'"}), 400
    # Security: allow subfolders, but prevent path traversal
    if '..' in book_filename_embed or book_filename_embed.startswith('/') or '\\' in book_filename_embed:
        logging.warning(f"Attempted path traversal in get_chunk_text: {book_filename_embed}")
        return jsonify({"error": "Invalid book filename"}), 400
    if not book_filename_embed.endswith('_embeded.json'):
        logging.warning(f"Invalid filename extension in get_chunk_text: {book_filename_embed}")
        return jsonify({"error": f"Invalid book filename extension: {book_filename_embed}"}), 400
    chunk_json_rel = book_filename_embed.replace('_embeded.json', '.json')
    original_path = os.path.join('chunks_json', chunk_json_rel)
    chunk_text = None
    total_chunks = 0
    if os.path.isfile(original_path):
        try:
            with open(original_path, 'r', encoding='utf-8') as f:
                original_chunks_data = json.load(f)
            total_chunks = len(original_chunks_data)
            for chunk_data in original_chunks_data:
                if chunk_data.get('chunk_index') == chunk_index:
                    chunk_text = chunk_data.get('text')
                    break
        except Exception as e:
            logging.error(f"Error reading or processing chunk file {original_path}: {e}")
            return jsonify({"error": "Failed to read chunk data"}), 500
    else:
        logging.error(f"Original chunk file not found: {original_path}")
        return jsonify({"error": "Chunk data file not found"}), 404
    if chunk_text is not None:
        logging.info(f"Retrieved text for Chunk {chunk_index} from {chunk_json_rel}")
        return jsonify({
            "text": chunk_text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        })
    else:
        logging.warning(f"Chunk index {chunk_index} not found in {original_path}")
        return jsonify({"error": f"Chunk index {chunk_index} not found", "total_chunks": total_chunks}), 404

if __name__ == '__main__':
    # Ensure the embeddings directory exists
    if not os.path.isdir(EMBEDDINGS_DIR):
        logging.error(f"Embeddings directory '{EMBEDDINGS_DIR}' not found. Please create it and add embedding files.")
    else:
        logging.info(f"Embeddings directory found at: {os.path.abspath(EMBEDDINGS_DIR)}")
    app.run(debug=True) 