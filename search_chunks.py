import asyncio
import cohere
import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI # Added AsyncOpenAI
import google.generativeai as genai # Added for Gemini
import logging
import re
from pathlib import Path # Added for easier path manipulation
from typing import Any, Tuple # For type hinting

# --- Configuration ---
COHERE_MODEL_NAME = "embed-v4.0"
INPUT_TYPE_QUERY = "search_document"
EMBEDDING_TYPE = "float"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
TOP_N_CHUNKS_PER_BOOK = 15 # Number of chunks per book for stage 1
MAX_LLM_TOKENS = 8000
CHUNKS_JSON_DIR = Path("chunks_json") # Define path to chunks
EMBEDDINGS_DIR = Path("embedded_books") # Define path to embeddings

# --- LLM Configuration ---
# Set the provider and model name here to switch LLMs.
# Ensure the corresponding API key is set in your .env file.
LLM_PROVIDER = "deepseek"  # Options: "deepseek", "openai", "gemini", "openrouter"
LLM_MODEL_NAME = "deepseek-chat"
# --- Available Model Options (User Specified) ---
# deepseek:
#   - deepseek-chat
# openai:
#   - gpt-4.1-mini
#   - gpt-4.1-nano
# gemini:
#   - gemini-2.5-flash-preview-04-17
#   - gemini-2.0-flash
#   - gemini-2.0-flash-lite
# openrouter:
#   - qwen/qwen3-235b-a22b


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load API Keys ---
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Debug: Print loaded OpenAI Key ---
print(f"DEBUG: Loaded OPENAI_API_KEY: {OPENAI_API_KEY}")
# -------------------------------------

# Validate API Keys based on selected provider
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file.")

API_KEY_MAP = {
    "deepseek": DEEPSEEK_API_KEY,
    "openai": OPENAI_API_KEY,
    "gemini": GEMINI_API_KEY,
    "openrouter": OPENROUTER_API_KEY,
}

SELECTED_API_KEY = API_KEY_MAP.get(LLM_PROVIDER)
if not SELECTED_API_KEY:
    raise ValueError(f"{LLM_PROVIDER.upper()}_API_KEY not found in .env file for selected provider.")


# --- Helper Function to Load Chunk Text ---
def load_chunk_text(embeddings_file_path: str, chunk_index: int) -> str | None:
    """Loads the text for a specific chunk from its JSON file, using the embedding file path to preserve subfolders."""
    # Replace 'embedded_books' with 'chunks_json' and '_embeded.json' with '.json'
    chunk_file_path = (
        Path(str(embeddings_file_path).replace('embedded_books', 'chunks_json').replace('_embeded.json', '.json'))
    )
    try:
        if not chunk_file_path.is_file():
            logging.warning(f"Original chunk file not found: {chunk_file_path}")
            return None
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        for chunk in chunks_data:
            if chunk.get('chunk_index') == chunk_index:
                return chunk.get('text')
        logging.warning(f"Chunk index {chunk_index} not found in {chunk_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading chunk text from {chunk_file_path} for index {chunk_index}: {e}", exc_info=True)
        return None


# --- LLM Client Initialization ---
async def _initialize_llm_client(provider: str, api_key: str) -> Tuple[Any | None, str | None]:
    """Initializes the appropriate async LLM client based on the provider."""
    logging.info(f"Initializing LLM client for provider: {provider}")
    if provider == "gemini":
        try:
            genai.configure(api_key=api_key)
            # Model is specified later during the call for Gemini
            client = genai.GenerativeModel(LLM_MODEL_NAME) # Pass model name here
            return client, None
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            return None, f"Failed to initialize Gemini client: {e}"
    elif provider in ["openai", "deepseek", "openrouter"]:
        base_url = None
        if provider == "deepseek":
            base_url = "https://api.deepseek.com"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        # OpenAI uses the default base_url

        try:
            # Use context manager for OpenAI-compatible clients
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            # We return the client instance, the caller should manage its lifecycle (e.g., async with)
            return client, None
        except Exception as e:
            logging.error(f"Failed to initialize {provider} client: {e}", exc_info=True)
            return None, f"Failed to initialize {provider} client: {e}"
    else:
        logging.error(f"Unsupported LLM provider: {provider}")
        return None, f"Unsupported LLM provider: {provider}"

# --- Abstracted LLM Call ---
async def _get_llm_completion(
    client: Any,
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0
) -> str | None:
    """Gets completion from the initialized LLM client, handling provider differences."""
    logging.debug(f"Requesting LLM completion. Provider: {provider}, Model: {model_name}")
    try:
        if provider == "gemini":
            # Gemini uses a different structure
            full_prompt = f"{system_prompt}{user_prompt}" # Combine prompts for Gemini
            logging.debug(f"Gemini Full Prompt:{full_prompt}")
            response = await client.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
                generation_config=genai.types.GenerationConfig(
                    # candidate_count=1, # Default is 1
                    # stop_sequences=['...'], # Optional
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 completion_text = response.candidates[0].content.parts[0].text.strip()
                 logging.debug(f"Gemini Raw Response Text:{completion_text}")
                 return completion_text
            else:
                 logging.error(f"Gemini returned unexpected structure. Response: {response}")
                 # Log safety ratings if available
                 if response and response.prompt_feedback:
                     logging.error(f"Gemini Prompt Feedback: {response.prompt_feedback}")
                 if response and response.candidates and response.candidates[0].finish_reason:
                     logging.error(f"Gemini Finish Reason: {response.candidates[0].finish_reason}")
                 if response and response.candidates and response.candidates[0].safety_ratings:
                    logging.error(f"Gemini Safety Ratings: {response.candidates[0].safety_ratings}")

                 return None

        elif provider in ["openai", "deepseek", "openrouter"]:
             # Uses OpenAI-compatible API
             logging.debug(f"OpenAI-like System Prompt:{system_prompt}")
             logging.debug(f"OpenAI-like User Prompt:{user_prompt}")
             messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
             ]
             # Check if the client is the AsyncOpenAI instance itself or needs context management
             if isinstance(client, AsyncOpenAI):
                 # If we passed the client instance directly, call it directly
                 completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                 )
             else:
                 # Should not happen with current structure, but as safeguard
                 logging.error("Invalid client type for OpenAI-like provider.")
                 return None

             # Check completion structure before accessing choices
             if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
                 raw_response_str = str(completion) if completion else "None"
                 logging.error(f"[{provider}] LLM returned unexpected structure. Raw Response: {raw_response_str[:1000]}...")
                 if isinstance(raw_response_str, str) and "<!doctype html" in raw_response_str.lower():
                      logging.error(f"[{provider}] LLM response appears to be HTML.")
                 return None
             else:
                 completion_text = completion.choices[0].message.content.strip()
                 logging.debug(f"[{provider}] Raw Response Content:{completion_text}")
                 return completion_text
        else:
             logging.error(f"Unsupported provider in _get_llm_completion: {provider}")
             return None

    except Exception as e:
        # Log specific API errors if available (e.g., from OpenAI library)
        # Note: Specific error handling might differ between libraries
        error_message = f"Error during LLM API call for {provider}: {e}"
        # Attempt to get more details if it's a known API error type (pseudo-code)
        # if hasattr(e, 'response') and e.response is not None:
        #     error_message += f" | Status Code: {e.response.status_code}"
        #     try:
        #        error_message += f" | Body: {await e.response.text()}" # Be careful with large bodies
        #     except: pass # Ignore if body cannot be read

        logging.error(error_message, exc_info=True)
        return None


# --- Stage 1: Process Single Book (Async) ---
async def _process_single_book(query: str, embeddings_file_path: str, top_n: int, co_client: cohere.AsyncClient, llm_client: Any, llm_provider: str, llm_model: str) -> dict | None:
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
            chunk_text = load_chunk_text(embeddings_file_path, chunk_index)
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
    logging.info(
        f"[Book: {book_filename_base}] Asking LLM (Stage 1 - Provider: {llm_provider}) to choose best chunk."
    )
    system_prompt_stage1 = (
        f"""
You are a helpful assistant analyzing text chunks from the book "{book_name_display}".
Your goal is to find the *single* chunk that is *most relevant* to the user's query.

1. Think step-by-step in a REASONING block about which chunk best matches the query.
2. After the reasoning, output *only* the following line:
   Final Answer: chunk <XX or None>
   Where XX is the single best chunk index. If no chunk is relevant, output "None".
   Absolutely no other text after this line. Do not use quotes.
"""
    )
    user_prompt_stage1 = (
        f"""
Query:
{query}

Potential Text Chunks from "{book_name_display}":
{top_chunks_for_llm}
Remember:
- Start with REASONING:.
- Finish with one line: Final Answer: chunk <XX or None>
"""
    )
    llm_response_content = None
    try:
        # Use the abstracted function
        llm_response_content = await _get_llm_completion(
            client=llm_client,
            provider=llm_provider,
            model_name=llm_model,
            system_prompt=system_prompt_stage1,
            user_prompt=user_prompt_stage1,
            max_tokens=MAX_LLM_TOKENS, # Use existing constant
            temperature=0
        )
        # Check if content was retrieved before logging success
        if llm_response_content is not None:
            logging.debug(f"[Book: {book_filename_base}] Stage 1 LLM Raw Response Content:{llm_response_content}")
        else:
             logging.error(f"[Book: {book_filename_base}] Stage 1 LLM call using _get_llm_completion failed or returned None.")
             # No need to log structure errors here, _get_llm_completion already did

    except Exception as e:
        # This catch block might be redundant if _get_llm_completion handles all exceptions
        # but kept as a safeguard during refactoring.
        logging.error(f"[Book: {book_filename_base}] Unexpected error calling _get_llm_completion for Stage 1: {e}", exc_info=True)
        llm_response_content = None # Ensure it's None

    # Write Stage 1 LLM output to file
    try:
        with open("last_llm_output.txt", "a", encoding="utf-8") as f:
            f.write(f"--- Stage 1 LLM Response ---\n")
            f.write(f"Book: {book_filename_base}\n")
            f.write(f"Query: {query}\n\n")
            f.write(f"{llm_response_content}\n\n")
    except Exception as e:
        logging.error(f"[Book: {book_filename_base}] Error writing Stage 1 LLM response to file: {e}", exc_info=True)

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
    # Clear last LLM output file for this query
    try:
        with open("last_llm_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n\n")
    except Exception as e:
        logging.error(f"Error clearing last_llm_output.txt: {e}", exc_info=True)

    # Create clients based on configuration
    # Cohere client remains the same
    cohere_client = cohere.AsyncClient(api_key=COHERE_API_KEY)

    # Initialize the selected LLM client
    llm_client_instance, init_error = await _initialize_llm_client(LLM_PROVIDER, SELECTED_API_KEY)

    if init_error or llm_client_instance is None:
        logging.error(f"Failed to initialize LLM client: {init_error}. Aborting search.")
        # Close cohere client if LLM fails to init
        await cohere_client.close()
        return []

    # We need to handle closing the clients properly.
    # Using 'async with' is ideal for AsyncOpenAI. Gemini doesn't require explicit close.
    # We'll wrap the main logic in try/finally to ensure cohere_client is closed.
    # If the llm_client is AsyncOpenAI, we should ideally use 'async with' around its usage.
    # Let's manage the AsyncOpenAI client with 'async with' if it's the one initialized.

    final_result_list = []
    try:
        # Number of books and log start
        num_books = len(embeddings_file_paths)
        logging.info(f"Initiating 2-stage search for query: '{query}' across {num_books} books using {LLM_PROVIDER}/{LLM_MODEL_NAME}.")

        # Define the main processing logic within a context manager for the LLM client if applicable
        async def run_search_stages(current_llm_client):
            # --- Stage 1: Run per-book processing concurrently ---
            stage1_tasks = [
                _process_single_book(
                    query,
                    file_path,
                    top_n_per_book,
                    cohere_client, # Pass cohere client
                    current_llm_client, # Pass the initialized (and potentially context-managed) LLM client
                    LLM_PROVIDER, # Pass provider info
                    LLM_MODEL_NAME # Pass model name info
                )
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
                return [] # Return empty list from inner function

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
                    "percent_into_book": final_result.get('percent_into_book'),
                    "audio_timestamp": final_result.get('audio_timestamp'),
                }
                return [final_result_formatted] # Return list from inner function
            elif num_books == 1: # No successful candidate for the single book
                logging.warning("Single book search failed to produce a candidate.")
                return [] # Return empty list from inner function


            # --- Stage 2: Final LLM Refinement ---
            logging.info(f"Starting Stage 2: Asking LLM ({LLM_PROVIDER}) to choose the best chunk among Stage 1 winners.")

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
                # Corrected f-string for newline characters
                stage2_input_chunks += f"-- {candidate_id}\n-- Book: {book_name}\n-- Chunk Index: {chunk_index}\n-- Text:\n{chunk_text}\n\n"
                # Store mapping from this ID back to the original candidate data
                candidate_map[candidate_id] = candidate


            if not candidate_map: # Check if any valid candidates remain for Stage 2
                logging.error("No valid candidates available for Stage 2 LLM refinement.")
                return [] # Return empty list from inner function

            system_prompt_stage2 = """
                You are a helpful assistant skilled in literary analysis.
                You will be given a user query and several candidate text chunks, each identified by a
                Candidate ID (e.g., "Candidate 1"), Book Name, and Chunk Index.
                Your task is to determine which *single* candidate chunk is the *overall best* match for
                the query.

                1. Think step-by-step in a REASONING block about which candidate chunk best addresses the
                   query. Consider relevance, detail, and context.
                2. After the reasoning, output *only* the following line:
                   Final Answer: <Candidate ID>
                   Where <Candidate ID> is the identifier (e.g., "Candidate 1", "Candidate 2") of the
                   single best chunk.
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
                # Use the abstracted function again for Stage 2
                llm_response_content_s2 = await _get_llm_completion(
                    client=current_llm_client, # Pass the LLM client
                    provider=LLM_PROVIDER,
                    model_name=LLM_MODEL_NAME,
                    system_prompt=system_prompt_stage2,
                    user_prompt=user_prompt_stage2,
                    max_tokens=MAX_LLM_TOKENS,
                    temperature=0
                )
                if llm_response_content_s2 is not None:
                    # Logged as INFO before, keeping consistency
                    logging.info(
                        f"Stage 2 LLM Raw Response Content:\n{llm_response_content_s2}"
                    )
                else:
                    logging.error(
                            "Stage 2 LLM call using _get_llm_completion failed or returned None."
                        )


            except Exception as e:
                # Safeguard catch
                logging.error(f"Unexpected error calling _get_llm_completion for Stage 2: {e}", exc_info=True)
                llm_response_content_s2 = None

            # Write Stage 2 LLM output to file
            try:
                with open("last_llm_output.txt", "a", encoding="utf-8") as f:
                    # Corrected f-strings for newline characters
                    f.write(f"--- Stage 2 LLM Response ---\n")
                    f.write(f"Query: {query}\n\n")
                    f.write(f"{llm_response_content_s2}\n\n") # Write content which might be None
            except Exception as e:
                logging.error(f"Error writing Stage 2 LLM response to file: {e}", exc_info=True)


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
                # Corrected f-string formatting for quotes
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
                return [final_result_formatted] # Return list from inner function
            else:
                logging.error("Could not determine a final result after Stage 2 processing and fallback.")
                return [] # Return empty list from inner function

        # Execute the search logic, managing the LLM client context if needed
        if isinstance(llm_client_instance, AsyncOpenAI):
            # Use async with for OpenAI-compatible clients
            logging.debug("Using async context manager for OpenAI-compatible client.")
            async with llm_client_instance as client_context:
                final_result_list = await run_search_stages(client_context)
        elif LLM_PROVIDER == "gemini":
             # Gemini client doesn't need explicit async with or close
             logging.debug("Using Gemini client directly (no context manager needed).")
             final_result_list = await run_search_stages(llm_client_instance)
        else:
             # Fallback or should not happen if _initialize_llm_client is correct
             logging.error(f"LLM client type ({type(llm_client_instance)}) not handled for context management.")
             # Attempt to run anyway, assuming the client doesn't need specific closing
             final_result_list = await run_search_stages(llm_client_instance)


    except Exception as e:
        logging.error(f"An unexpected error occurred during the main search process: {e}", exc_info=True)
        final_result_list = [] # Ensure empty list on major error
    finally:
        # Ensure Cohere client is always closed
        if cohere_client:
            logging.info("Cohere client does not require explicit close.")
        # Note: AsyncOpenAI client is closed by its 'async with' block if used.
        # Gemini client does not require explicit closing.

    return final_result_list