import cohere
import numpy as np
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
CHUNKS_JSON_DIR = "chunks_json"
EMBEDDED_BOOKS_DIR = "embedded_books"
SKIP_SERIES = ["First Law", "Gods_of_Blood_and_Powder", "Powder_Mage_Trilogy"]
MODEL_NAME = "embed-v4.0"
INPUT_TYPE = "search_document"
EMBEDDING_TYPE = "float" # Options: float, int8, uint8, binary, ubinary
BATCH_SIZE = 96 # Cohere API batch size limit

# --- Load API Key ---
# Loads API key from .env file in the same directory
load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Create a .env file with COHERE_API_KEY.")

# --- Initialize Cohere Client ---
try:
    # Use cohere.ClientV2 for newer versions (5.x+)
    co = cohere.ClientV2(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Cohere client: {e}")
    exit(1)

# --- Helper Function for Output Filename ---
def generate_output_filename(series, input_filename):
    """Generates the output JSON filename for embeddings in the correct output directory."""
    base_name = os.path.basename(input_filename).replace(".json", "")
    output_dir = os.path.join(EMBEDDED_BOOKS_DIR, series)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_name}_embeded.json")

# --- Embedding Function (handles one batch) ---
def get_embeddings_batch(text_batch, model_name, input_type, embedding_type, cohere_client):
    try:
        response = cohere_client.embed(
            texts=text_batch,
            model=model_name,
            input_type=input_type,
            embedding_types=[embedding_type],
        )
        embeddings = getattr(response.embeddings, embedding_type, None)
        if embeddings is None:
            print(f"Error: Could not retrieve '{embedding_type}' embeddings from the API response.")
            return None # Indicate error
        if len(embeddings) != len(text_batch):
            print(f"Error: Mismatch between number of texts ({len(text_batch)}) and received embeddings ({len(embeddings)}).")
            return None # Indicate error
        return embeddings
    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        return None # Indicate error

# --- Main Processing Loop ---
def main():
    processed_files = 0
    failed_files = 0

    # Discover all series (subfolders) in chunks_json
    all_series = [s for s in os.listdir(CHUNKS_JSON_DIR) if os.path.isdir(os.path.join(CHUNKS_JSON_DIR, s))]
    # Only process series that are not in SKIP_SERIES
    to_process_series = [s for s in all_series if s not in SKIP_SERIES]

    for series in to_process_series:
        series_path = os.path.join(CHUNKS_JSON_DIR, series)
        print(f"\n=== Processing series: {series} ===")
        for book_file in os.listdir(series_path):
            if not book_file.endswith('.json'):
                continue
            input_json_file = os.path.join(series_path, book_file)
            output_json_file = generate_output_filename(series, book_file)
            print(f"\n--- Processing: {input_json_file} ---")

            # --- Load Data for current file ---
            try:
                with open(input_json_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
            except FileNotFoundError:
                print(f"Error: Input file '{input_json_file}' not found. Skipping.")
                failed_files += 1
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{input_json_file}'. Skipping.")
                failed_files += 1
                continue
            except Exception as e:
                print(f"An unexpected error occurred while reading '{input_json_file}': {e}. Skipping.")
                failed_files += 1
                continue

            # --- Prepare Texts for Embedding ---
            texts_to_embed = [item['text'] for item in chunks_data if isinstance(item, dict) and 'text' in item and item['text']]

            if not texts_to_embed:
                print(f"No valid text chunks found in '{input_json_file}'. Skipping.")
                failed_files += 1
                continue

            # --- Generate Embeddings in Batches ---
            print(f"Generating embeddings for {len(texts_to_embed)} chunks using model '{MODEL_NAME}'...")
            all_embeddings = []
            embeddings_failed = False
            for start in range(0, len(texts_to_embed), BATCH_SIZE):
                batch = texts_to_embed[start:start+BATCH_SIZE]
                print(f"  Embedding batch {start//BATCH_SIZE + 1} ({start} to {start+len(batch)-1})...")
                batch_embeddings = get_embeddings_batch(batch, MODEL_NAME, INPUT_TYPE, EMBEDDING_TYPE, co)
                if batch_embeddings is None:
                    print(f"Error embedding batch for '{input_json_file}'. Aborting embedding for this file.")
                    embeddings_failed = True
                    break # Stop processing batches for this file
                all_embeddings.extend(batch_embeddings)

            if embeddings_failed:
                failed_files += 1
                continue # Move to the next file

            if len(all_embeddings) != len(texts_to_embed):
                print(f"Error: Total embeddings received ({len(all_embeddings)}) does not match number of texts ({len(texts_to_embed)}) for '{input_json_file}'. Skipping save.")
                failed_files += 1
                continue

            print("Embeddings generated successfully.")

            # --- Combine Original Data with Embeddings ---
            output_data = []
            embedding_index = 0
            valid_chunk_count = 0
            for i, item in enumerate(chunks_data):
                # Ensure the item is a dict and has a non-empty text field (matches texts_to_embed)
                if isinstance(item, dict) and 'text' in item and item['text']:
                    valid_chunk_count += 1
                    if embedding_index < len(all_embeddings):
                        output_item = item.copy()
                        # Ensure embedding is converted to list for JSON serialization
                        embedding_list = all_embeddings[embedding_index]
                        if isinstance(embedding_list, np.ndarray):
                             embedding_list = embedding_list.tolist()
                        output_item['embedding'] = embedding_list
                        output_data.append(output_item)
                        embedding_index += 1
                    else:
                         # This case should ideally not happen if checks above are correct, but added for safety
                        print(f"Warning: Missing embedding for valid text item index {i} in {input_json_file}. Discrepancy detected.")

            if valid_chunk_count != len(all_embeddings):
                 print(f"Warning: Number of valid text chunks ({valid_chunk_count}) doesn't match embeddings ({len(all_embeddings)}) for {input_json_file}. Saving partial results.")

            # --- Save Results ---
            if not output_data:
                print(f"No data to save for '{input_json_file}'. Skipping save.")
                failed_files += 1
                continue

            print(f"Saving combined data and embeddings to '{output_json_file}'...")
            try:
                with open(output_json_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
                print(f"Successfully saved embeddings to '{output_json_file}'.")
                processed_files += 1
            except IOError as e:
                print(f"Error writing output file '{output_json_file}': {e}. Skipping.")
                failed_files += 1
            except Exception as e:
                print(f"An unexpected error occurred while saving '{output_json_file}': {e}. Skipping.")
                failed_files += 1

    print(f"\n--- Script Finished. Processed {processed_files} files successfully. Failed/Skipped {failed_files} files. ---")

if __name__ == "__main__":
    main() 