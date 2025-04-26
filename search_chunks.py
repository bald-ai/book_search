import asyncio
import cohere
import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI # Added AsyncOpenAI
import logging
import re
from pathlib import Path # Added for easier path manipulation

# --- Configuration ---
COHERE_MODEL_NAME = "embed-v4.0"
INPUT_TYPE_QUERY = "search_document"
EMBEDDING_TYPE = "float"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
TOP_N_CHUNKS_PER_BOOK = 15 # Number of chunks per book for stage 1
MAX_LLM_TOKENS = 8000
CHUNKS_JSON_DIR = Path("chunks_json") # Define path to chunks
EMBEDDINGS_DIR = Path("embedded_books") # Define path to embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load API Keys ---
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file.")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file.")

# --- Initialize Async Clients ---
try:
    # Use AsyncClient for Cohere
    co_async = cohere.AsyncClient(api_key=COHERE_API_KEY)
    logging.info("Cohere AsyncClient initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Cohere AsyncClient: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize Cohere AsyncClient: {e}")

try:
    # Use AsyncOpenAI for DeepSeek
    deepseek_async_client = AsyncOpenAI(
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_API_KEY,
    )
    logging.info("DeepSeek AsyncOpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing DeepSeek AsyncOpenAI client: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize DeepSeek AsyncOpenAI client: {e}")

# --- Helper Function to Load Chunk Text ---
def load_chunk_text(book_filename_base: str, chunk_index: int) -> str | None:
    """Loads the text for a specific chunk from its JSON file."""
    json_filename = f"{book_filename_base}.json"
    file_path = CHUNKS_JSON_DIR / json_filename
    try:
        if not file_path.is_file():
            logging.warning(f"Original chunk file not found: {file_path}")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        for chunk in chunks_data:
            if chunk.get('chunk_index') == chunk_index:
                return chunk.get('text')
        logging.warning(f"Chunk index {chunk_index} not found in {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading chunk text from {file_path} for index {chunk_index}: {e}", exc_info=True)
        return None


# --- Stage 1: Process Single Book (Async) ---
async def _process_single_book(query: str, embeddings_file_path: str, top_n: int, co_client: cohere.AsyncClient, llm_client: AsyncOpenAI) -> dict | None:
    """Loads data, performs semantic search, and LLM refinement for ONE book."""
    file_path = Path(embeddings_file_path)
    book_filename_base = file_path.stem.replace('_embeded', '') # e.g., "promise_of_blood"
    logging.info(f"[Book: {book_filename_base}] Starting Stage 1 processing.")

    # --- Load Data ---
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        if not book_data:
            logging.warning(f"[Book: {book_filename_base}] Embeddings file is empty.")
            return None
        # Assume book_name is consistent within the file, take from first item
        book_name_display = book_data[0].get('book_name', book_filename_base.replace('_', ' ').title())
        doc_embeddings = np.array([item['embedding'] for item in book_data])
        logging.info(f"[Book: {book_filename_base}] Loaded {len(book_data)} chunks. Embedding shape: {doc_embeddings.shape}")
    except Exception as e:
        logging.error(f"[Book: {book_filename_base}] Error loading data: {e}", exc_info=True)
        return None

    # --- Semantic Search ---
    query_embedding = None
    try:
        response = await co_client.embed(
            texts=[query],
            model=COHERE_MODEL_NAME,
            input_type=INPUT_TYPE_QUERY,
            embedding_types=[EMBEDDING_TYPE],
        )
        # Check response structure before accessing attributes
        if not hasattr(response, 'embeddings') or not hasattr(response.embeddings, EMBEDDING_TYPE):
            logging.error(f"[Book: {book_filename_base}] Unexpected Cohere response structure. Response: {response}")
            return None
        query_embedding = getattr(response.embeddings, EMBEDDING_TYPE, None)

    except Exception as e:
        # Catch potential connection errors or other issues during the API call itself
        logging.error(f"[Book: {book_filename_base}] Error during Cohere embed API call: {e}", exc_info=True)
        return None # Exit if API call failed

    if not query_embedding:
        logging.error(f"[Book: {book_filename_base}] Could not get query embedding after API call.")
        return None

    try:
        query_emb_np = np.array(query_embedding[0])
        scores = np.dot(query_emb_np, doc_embeddings.T)
        actual_top_n = min(top_n, len(book_data))
        top_n_indices = np.argsort(scores)[::-1][:actual_top_n]

        semantic_results_data = []
        top_chunks_for_llm = ""
        logging.info(f"[Book: {book_filename_base}] Top {actual_top_n} semantic matches:")
        for i, idx in enumerate(top_n_indices):
            item_data = book_data[idx]
            chunk_index = item_data.get('chunk_index', -1)
            score = float(scores[idx])
            # Load text from chunks_json for the LLM
            chunk_text = load_chunk_text(book_filename_base, chunk_index)
            if chunk_text is None:
                logging.warning(f"[Book: {book_filename_base}] Could not load text for chunk {chunk_index} for LLM.")
                text_for_llm = item_data.get('text', '') # Fallback to text in embedding file
            else:
                text_for_llm = chunk_text

            logging.info(f"  Rank {i}, Chunk {chunk_index}, Score: {score:.4f}")
            top_chunks_for_llm += f"-- Chunk Index: {chunk_index}\\n-- Text:\\n{text_for_llm}\\n\\n"

            # Store data including text from embedding file (for potential final return)
            result_item = {
                "book_name": book_name_display, # Store display name
                "book_filename_base": book_filename_base, # Store base name for later lookups
                "chunk_index": chunk_index,
                "text_from_embedding_file": item_data.get('text', ''), # Original text
                "score": score,
                "rank": i, # Rank within this book's semantic search
                "embeddings_file_path": embeddings_file_path, # Keep track of original file path
                # Include other metadata if needed by the frontend/caller from embedding file
                # Ensure these keys exist or use .get() with defaults
                "percent_into_book": item_data.get('percent_into_book'),
                "audio_timestamp": item_data.get('audio_timestamp'),
            }
            semantic_results_data.append(result_item)

        if not semantic_results_data:
            logging.warning(f"[Book: {book_filename_base}] Semantic search yielded no results.")
            return None

    except Exception as e:
        logging.error(f"[Book: {book_filename_base}] Error processing semantic search results: {e}", exc_info=True)
        return None

    # --- LLM Refinement (Stage 1) ---
    logging.info(f"[Book: {book_filename_base}] Asking LLM (Stage 1) to choose best chunk.")
    system_prompt_stage1 = f"""
You are a helpful assistant analyzing text chunks from the book "{book_name_display}".
Your goal is to find the *single* chunk that is *most relevant* to the user's query.

1. Think step-by-step in a REASONING block about which chunk best matches the query.
2. After the reasoning, output *only* the following line:
   Final Answer: chunk <XX or None>
   Where XX is the single best chunk index. If no chunk is relevant, output "None".
   Absolutely no other text after this line. Do not use quotes.
"""
    user_prompt_stage1 = f"""
Query:
{query}

Potential Text Chunks from "{book_name_display}":
{top_chunks_for_llm}
Remember:
- Start with REASONING:.
- Finish with one line: Final Answer: chunk <XX or None>
"""
    llm_response_content = None
    try:
        completion = await llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt_stage1},
                {"role": "user", "content": user_prompt_stage1}
            ],
            max_tokens=MAX_LLM_TOKENS,
            temperature=0
        )
        # Check completion structure before accessing choices
        if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
            # Attempt to log the raw completion object if possible
            raw_response_str = str(completion) if completion else "None"
            logging.error(f"[Book: {book_filename_base}] Stage 1 LLM returned unexpected structure. Raw Response: {raw_response_str[:1000]}...") # Log first 1k chars
            # Try to check if it's an HTML error
            if isinstance(raw_response_str, str) and "<!doctype html" in raw_response_str.lower():
                 logging.error(f"[Book: {book_filename_base}] Stage 1 LLM response appears to be HTML.")
            llm_response_content = None # Explicitly set to None
        else:
            llm_response_content = completion.choices[0].message.content.strip()
            logging.debug(f"[Book: {book_filename_base}] Stage 1 LLM Raw Response Content:\n{llm_response_content}")

    except Exception as e:
        logging.error(f"[Book: {book_filename_base}] Error during Stage 1 LLM API call: {e}", exc_info=True)
        # Attempt to log the exception details, maybe it contains the response?
        # We cannot reliably get the raw response body from the exception object itself in most HTTP libraries
        llm_response_content = None # Ensure it's None

    # --- Parse Stage 1 LLM Response (only if content was successfully retrieved) ---
    final_chunk_index = None
    if llm_response_content:
        try:
            match = re.search(r"Final Answer:\s*chunk\s*(\d+|None)", llm_response_content, re.IGNORECASE | re.MULTILINE)
            if match:
                selected = match.group(1)
                if selected.lower() == 'none':
                    final_chunk_index = None
                    logging.info(f"[Book: {book_filename_base}] Stage 1 LLM selected None.")
                else:
                    try:
                        final_chunk_index = int(selected)
                        logging.info(f"[Book: {book_filename_base}] Stage 1 LLM selected Chunk Index: {final_chunk_index}")
                    except ValueError:
                        logging.warning(f"[Book: {book_filename_base}] Stage 1 LLM failed to parse index from: '{selected}'")
            else:
                logging.warning(f"[Book: {book_filename_base}] Stage 1 LLM could not find 'Final Answer: chunk' line in response: {llm_response_content[:500]}...")
        except Exception as parse_error:
             logging.error(f"[Book: {book_filename_base}] Error parsing Stage 1 LLM response: {parse_error}. Response: {llm_response_content[:500]}...", exc_info=True)
             final_chunk_index = None # Ensure fallback on parsing error
    else:
         logging.warning(f"[Book: {book_filename_base}] Skipping Stage 1 LLM response parsing due to previous error or empty response.")

    # --- Find Selected Chunk Data ---
    selected_item = None
    if final_chunk_index is not None:
        for item in semantic_results_data:
            if item.get('chunk_index') == final_chunk_index:
                selected_item = item
                logging.info(f"[Book: {book_filename_base}] Found Stage 1 LLM choice (Chunk {final_chunk_index}, Rank {item.get('rank')}).")
                break
        if not selected_item:
             logging.warning(f"[Book: {book_filename_base}] Stage 1 LLM chose Chunk {final_chunk_index}, but not found in top {actual_top_n} semantic results! Falling back.")
             # Fallback handled below

    # Fallback to top semantic if LLM failed or chose None/invalid
    if selected_item is None:
        if semantic_results_data:
            selected_item = semantic_results_data[0] # Fallback to top semantic match
            logging.info(f"[Book: {book_filename_base}] Falling back to top semantic result (Chunk {selected_item.get('chunk_index')}, Rank {selected_item.get('rank')}).")
        else:
            logging.error(f"[Book: {book_filename_base}] Cannot select result: LLM failed/None AND no semantic results.")
            return None

    return selected_item


# --- Main Search Function (Async) ---
async def search_book_chunks(query: str, embeddings_file_paths: list[str], top_n_per_book: int = TOP_N_CHUNKS_PER_BOOK) -> list[dict]:
    """
    Performs a two-stage search across multiple books concurrently.
    Stage 1: Finds the best chunk within each book using semantic search + LLM (parallel).
    Stage 2: Takes the best chunk from each book and uses LLM to find the overall best one.

    Args:
        query: The user's search query.
        embeddings_file_paths: List of paths to the books' embeddings JSON files.
        top_n_per_book: Number of semantic results per book for Stage 1 LLM.

    Returns:
        A list containing a single dictionary for the final LLM-selected chunk,
        including book_name, chunk_index, text (from embedding file), score, and rank
        (relative to its book's Stage 1 semantic search).
        Returns an empty list if no relevant chunk is found or an error occurs.
    """
    num_books = len(embeddings_file_paths)
    logging.info(f"Initiating 2-stage search for query: '{query}' across {num_books} books.")

    # --- Stage 1: Run per-book processing concurrently ---
    stage1_tasks = [
        _process_single_book(query, file_path, top_n_per_book, co_async, deepseek_async_client)
        for file_path in embeddings_file_paths
    ]
    stage1_results_raw = await asyncio.gather(*stage1_tasks, return_exceptions=True)

    # Filter out None results and exceptions
    stage1_successful_candidates = []
    for i, result in enumerate(stage1_results_raw):
        book_path = embeddings_file_paths[i]
        if isinstance(result, Exception):
            logging.error(f"Stage 1 processing for book {book_path} failed with exception: {result}", exc_info=result)
        elif result is None:
            logging.warning(f"Stage 1 processing for book {book_path} returned no candidate.")
        else:
            stage1_successful_candidates.append(result)

    if not stage1_successful_candidates:
        logging.warning("No successful candidates found after Stage 1 processing across all books.")
        return []

    logging.info(f"Stage 1 completed. Found {len(stage1_successful_candidates)} candidates from {num_books} books.")

    # If only one book was searched, its Stage 1 result is the final result.
    if num_books == 1 and stage1_successful_candidates: # Ensure candidate exists
        logging.info("Only one book searched, returning Stage 1 result directly.")
        final_result = stage1_successful_candidates[0]
        # Ensure the final result has the expected keys for app.py
        final_result_formatted = {
            "book_name": final_result.get("book_name"),
            "chunk_index": final_result.get("chunk_index"),
            "text": final_result.get("text_from_embedding_file"), # Text from embedding file
            "score": final_result.get("score"),
            "rank": final_result.get("rank"), # Rank within its book's semantic search
            # Add other fields if needed by app.py (ensure they exist in Stage 1 result)
            "percent_into_book": final_result.get('percent_into_book'),
            "audio_timestamp": final_result.get('audio_timestamp'),
        }
        return [final_result_formatted]
    elif num_books == 1: # No successful candidate for the single book
        logging.warning("Single book search failed to produce a candidate.")
        return []


    # --- Stage 2: Final LLM Refinement ---
    logging.info("Starting Stage 2: Asking LLM to choose the best chunk among Stage 1 winners.")

    stage2_input_chunks = ""
    candidate_map = {} # Store candidate data for lookup after LLM choice

    for i, candidate in enumerate(stage1_successful_candidates):
        book_name = candidate.get("book_name", "Unknown Book")
        chunk_index = candidate.get("chunk_index", -1)
        book_filename_base = candidate.get("book_filename_base")

        if book_filename_base is None or chunk_index == -1:
            logging.warning(f"Skipping candidate due to missing info: {candidate}")
            continue

        # Load text from chunks_json for Stage 2 LLM
        chunk_text = load_chunk_text(book_filename_base, chunk_index)
        if chunk_text is None:
            logging.warning(f"Could not load text for Stage 2 candidate: Book {book_name}, Chunk {chunk_index}. Using text from embedding file as fallback.")
            chunk_text = candidate.get('text_from_embedding_file', '') # Fallback

        # Use a unique identifier for the LLM prompt
        candidate_id = f"Candidate {i+1}"
        stage2_input_chunks += f"-- {candidate_id}\\n-- Book: {book_name}\\n-- Chunk Index: {chunk_index}\\n-- Text:\\n{chunk_text}\\n\\n"
        # Store mapping from this ID back to the original candidate data
        candidate_map[candidate_id] = candidate


    if not candidate_map: # Check if any valid candidates remain for Stage 2
        logging.error("No valid candidates available for Stage 2 LLM refinement.")
        return []

    system_prompt_stage2 = """
You are a helpful assistant skilled in literary analysis.
You will be given a user query and several candidate text chunks, each identified by a Candidate ID (e.g., "Candidate 1"), Book Name, and Chunk Index.
Your task is to determine which *single* candidate chunk is the *overall best* match for the query.

1. Think step-by-step in a REASONING block about which candidate chunk best addresses the query. Consider relevance, detail, and context.
2. After the reasoning, output *only* the following line:
   Final Answer: <Candidate ID>
   Where <Candidate ID> is the identifier (e.g., "Candidate 1", "Candidate 2") of the single best chunk.
   Absolutely no other text after this line. Do not use quotes.
"""
    user_prompt_stage2 = f"""
Query:
{query}

Candidate Text Chunks:
{stage2_input_chunks}
Remember:
- Start with REASONING:.
- Finish with one line: Final Answer: Candidate <N>
"""

    llm_response_content_s2 = None
    try:
        completion = await deepseek_async_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt_stage2},
                {"role": "user", "content": user_prompt_stage2}
            ],
            max_tokens=MAX_LLM_TOKENS,
            temperature=0
        )
        # Check completion structure before accessing choices
        if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
            raw_response_str = str(completion) if completion else "None"
            logging.error(f"Stage 2 LLM returned unexpected structure. Raw Response: {raw_response_str[:1000]}...")
            if isinstance(raw_response_str, str) and "<!doctype html" in raw_response_str.lower():
                 logging.error(f"Stage 2 LLM response appears to be HTML.")
            llm_response_content_s2 = None
        else:
            llm_response_content_s2 = completion.choices[0].message.content.strip()
            logging.info(f"Stage 2 LLM Raw Response Content:\n{llm_response_content_s2}") # Changed log level to info

    except Exception as e:
        logging.error(f"Error during Stage 2 LLM API call: {e}", exc_info=True)
        llm_response_content_s2 = None

    # --- Parse Stage 2 LLM Response ---
    final_candidate_id = None
    if llm_response_content_s2:
        try:
            # Make regex more robust to surrounding text or slight variations
            match = re.search(r"Final Answer:\s*(Candidate\s*\d+)", llm_response_content_s2, re.IGNORECASE | re.MULTILINE)
            if match:
                # Normalize the ID found (e.g., "Candidate  1" -> "Candidate 1")
                num = re.search(r'\d+', match.group(1))
                if num:
                    final_candidate_id = f"Candidate {num.group(0)}"
                    logging.info(f"Stage 2 LLM selected: {final_candidate_id}")
                else:
                    logging.warning(f"Could not extract number from Stage 2 LLM answer: {match.group(1)}")
            else:
                logging.warning(f"Stage 2 LLM could not find 'Final Answer: Candidate N' line. Falling back. Response: {llm_response_content_s2[:500]}...")
        except Exception as parse_error:
            logging.error(f"Error parsing Stage 2 LLM response: {parse_error}. Response: {llm_response_content_s2[:500]}...", exc_info=True)
            final_candidate_id = None
    else:
        logging.warning("Skipping Stage 2 LLM response parsing due to previous error or empty response.")

    # --- Select Final Result ---
    selected_item_data = None
    if final_candidate_id and final_candidate_id in candidate_map:
        selected_item_data = candidate_map[final_candidate_id]
        logging.info(f"Found Stage 2 LLM choice: {final_candidate_id} corresponding to Book '{selected_item_data.get('book_name')}', Chunk {selected_item_data.get('chunk_index')}")
    else:
        logging.warning("Stage 2 LLM choice invalid or parsing failed. Falling back based on Stage 1 rank/score.")
        if stage1_successful_candidates:
            stage1_successful_candidates.sort(key=lambda x: (x.get('rank', 999), -x.get('score', 0.0)))
            selected_item_data = stage1_successful_candidates[0]
            logging.info(f"Falling back to best Stage 1 candidate: Book '{selected_item_data.get('book_name')}', Chunk {selected_item_data.get('chunk_index')} (Rank {selected_item_data.get('rank')}, Score {selected_item_data.get('score', 0.0):.4f})")
        else:
             logging.error("Fallback failed: No successful stage 1 candidates were available for sorting.")

    # Format and return final result
    if selected_item_data:
        # Format the final result for app.py
        final_result_formatted = {
            "book_name": selected_item_data.get("book_name"),
            "chunk_index": selected_item_data.get("chunk_index"),
            "text": selected_item_data.get("text_from_embedding_file"),
            "score": selected_item_data.get("score"),
            "rank": selected_item_data.get("rank"),
            "percent_into_book": selected_item_data.get('percent_into_book'),
            "audio_timestamp": selected_item_data.get('audio_timestamp'),
        }
        return [final_result_formatted]
    else:
        logging.error("Could not determine a final result after Stage 2 processing and fallback.")
        return []