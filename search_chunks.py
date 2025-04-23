import cohere
import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI # Re-enabled for DeepSeek
import logging # Added for logging
import re # Import regex module

# --- Configuration ---
# EMBEDDINGS_FILE = "mistborn_full_embeddings.json" # Now passed as argument
COHERE_MODEL_NAME = "embed-v4.0"  # Must match embedding model
INPUT_TYPE_QUERY = "search_document"  # Must match embedding type
EMBEDDING_TYPE = "float"  # Must match embedding type
DEEPSEEK_MODEL_NAME = "deepseek-chat" # Re-enabled
TOP_N_CHUNKS = 15 # Number of chunks to send to LLM
MAX_LLM_TOKENS = 1000 # Max tokens for LLM response

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Load API Keys ---
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Re-enabled

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file.")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file.") # Re-enabled check

# --- Initialize Clients ---
try:
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    logging.info("Cohere client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Cohere client: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize Cohere client: {e}")

try:
    deepseek_client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_API_KEY,
    )
    logging.info("DeepSeek client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing DeepSeek client: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize DeepSeek client: {e}")

# --- Search Function ---
def search_book_chunks(query: str, embeddings_file_path: str, top_n: int = TOP_N_CHUNKS):
    """Searches chunks, refines with LLM, returns the single best chunk.

    Args:
        query: The user's search query.
        embeddings_file_path: Path to the book's embeddings JSON file.
        top_n: Number of semantic results to feed into the LLM.

    Returns:
        A list containing a single dictionary for the LLM-selected chunk,
        or an empty list if no relevant chunk is found or an error occurs.
    """
    # --- Load Embedded Data ---
    try:
        with open(embeddings_file_path, 'r', encoding='utf-8') as f:
            embedded_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Embeddings file not found: {embeddings_file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from: {embeddings_file_path}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error reading {embeddings_file_path}: {e}", exc_info=True)
        return []

    if not embedded_data:
        logging.warning(f"Embeddings file is empty: {embeddings_file_path}")
        return []

    # --- Extract Data ---
    try:
        # We need chunk_index and text for LLM prompt, and the full item for return
        doc_embeddings = np.array([item['embedding'] for item in embedded_data])
        # Store full original items mapped by their chunk index for easy lookup later
        docs_by_index = {item.get('chunk_index', i): item for i, item in enumerate(embedded_data)}
        doc_texts_for_prompt = [
             f"-- Chunk {item.get('chunk_index', i)} --\n{item['text']}\n\n"
             for i, item in enumerate(embedded_data)
        ]
    except KeyError as e:
        logging.error(f"Missing expected key '{e}' in {embeddings_file_path}")
        return []
    except Exception as e:
        logging.error(f"Error processing data from {embeddings_file_path}: {e}", exc_info=True)
        return []

    # --- Semantic Search Step ---
    try:
        response = co.embed(
            texts=[query],
            model=COHERE_MODEL_NAME,
            input_type=INPUT_TYPE_QUERY,
            embedding_types=[EMBEDDING_TYPE],
        )
        query_embedding = getattr(response.embeddings, EMBEDDING_TYPE, None)
        if not query_embedding:
            logging.error("Could not get embedding for the query.")
            return []
        query_emb_np = np.array(query_embedding[0])

        if query_emb_np.ndim != 1 or doc_embeddings.ndim != 2 or query_emb_np.shape[0] != doc_embeddings.shape[1]:
            logging.error(f"Embedding dimension mismatch. Query: {query_emb_np.shape}, Docs: {doc_embeddings.shape}")
            return []

        scores = np.dot(query_emb_np, doc_embeddings.T)
        actual_top_n = min(top_n, len(docs_by_index))
        # Get indices relative to the original embedded_data list
        top_n_indices_original = np.argsort(scores)[::-1][:actual_top_n]

        # Prepare data for LLM: Get the text chunks corresponding to top semantic matches
        top_chunks_for_llm = ""
        semantic_results_data = [] # Keep data of top N semantic matches
        logging.info(f"Top {actual_top_n} semantic matches (Indices: {top_n_indices_original}):")
        for i, original_index in enumerate(top_n_indices_original):
             chunk_index = embedded_data[original_index].get('chunk_index', original_index)
             text = embedded_data[original_index].get('text', '')
             score = float(scores[original_index])
             logging.info(f"  {i+1}. Chunk {chunk_index}, Score: {score:.4f}")
             top_chunks_for_llm += f"-- Chunk {chunk_index} --\n{text}\n\n"
             # Store the data needed if we fall back or need to retrieve the LLM choice
             semantic_results_data.append({
                 "chunk_index": chunk_index,
                 "text": text,
                 "score": score,
                 "percent_into_book": embedded_data[original_index].get('percent_into_book'),
                 "audio_timestamp": embedded_data[original_index].get('audio_timestamp'),
             })

        if not semantic_results_data:
             logging.warning("Semantic search returned no results to process.")
             return []

    except Exception as e:
        logging.error(f"Error during semantic search for query '{query}': {e}", exc_info=True)
        return []

    # --- LLM Refinement Step ---
    logging.info(f"Asking LLM ({DEEPSEEK_MODEL_NAME}) to choose best chunk for query: '{query}'")
    system_prompt = (
        "You are a helpful assistant skilled in literary analysis. "
        "Your task is to determine which single text chunk (identified by its 'Chunk Index') is the most relevant source "
        "for the event or detail described in the user's query, based ONLY on the provided text chunks. "
        "First, briefly reason step-by-step (Chain-of-Thought) considering how each chunk relates to the query. "
        "Then, conclude your response with the single most relevant chunk index on the last line, formatted EXACTLY as: Final Answer: chunk XX "
        "(where XX is the specific Chunk Index number from the list, e.g., 'Final Answer: chunk 142'). "
        "Only output the single best chunk index. Do not list multiple options. If no chunk seems relevant, state that clearly in the reasoning and output 'Final Answer: chunk None'."
    )
    user_prompt = f"Query:\n{query}\n\nPotential Text Chunks:\n{top_chunks_for_llm}\nAnalyze the query and the provided text chunks. Reason step-by-step which chunk index is the single most likely source for the event/detail in the query. Conclude with the final chunk index on a new line, formatted exactly as 'Final Answer: chunk XX' or 'Final Answer: chunk None'."

    try:
        completion = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_LLM_TOKENS,
            temperature=0 # Aim for deterministic output
        )
        llm_response = completion.choices[0].message.content.strip()
        logging.info(f"LLM Response:\n{llm_response}")

        # --- More Robust Parsing Logic ---
        final_chunk_index = None
        last_match_pos = -1
        found_none_indicator = False
        none_indicator_pos = -1

        if llm_response:
            # 1. Find all numerical matches ("Final Answer: chunk 123")
            pattern = r"final answer: chunk\s+(\d+).?" # Optional period at the end
            for match in re.finditer(pattern, llm_response, re.IGNORECASE):
                try:
                    potential_index = int(match.group(1))
                    # Store the index and position of the latest valid match
                    final_chunk_index = potential_index 
                    last_match_pos = match.start() # Record start position of this match
                    logging.info(f"Found potential match: Chunk {final_chunk_index} at position {last_match_pos}")
                except ValueError:
                    logging.warning(f"Regex matched number pattern, but failed to parse int from: {match.group(1)}. Match: '{match.group(0)}'.")
                    continue # Ignore invalid number formats

            # 2. Check for explicit "None" indicator after the last numerical match
            none_pattern = r"final answer: chunk none" 
            for none_match in re.finditer(none_pattern, llm_response, re.IGNORECASE): 
                found_none_indicator = True
                none_indicator_pos = none_match.start()
                logging.info(f"Found 'None' indicator at position {none_indicator_pos}")

            # 3. Decide based on findings
            if found_none_indicator and none_indicator_pos > last_match_pos:
                # If 'None' indicator appears *after* the last numerical match, respect it.
                logging.info("Found 'None' indicator after the last numerical match. Setting index to None.")
                final_chunk_index = None 
            elif final_chunk_index is None and not found_none_indicator:
                 # If no numerical match found and no 'None' found either, log warning.
                 logging.warning(f"Could not find a valid 'Final Answer: chunk XX' or 'Final Answer: chunk None' pattern in LLM response. Response:\n{llm_response}")
                 # final_chunk_index remains None, leading to fallback

        else:
            logging.warning("LLM response was empty. Falling back.")
            # final_chunk_index remains None

        # --- Prepare final result (fallback logic remains the same) ---
        if final_chunk_index is not None:
            chosen_chunk_data = next((res for res in semantic_results_data if res['chunk_index'] == final_chunk_index), None)
            if chosen_chunk_data:
                 logging.info(f"LLM chose Chunk Index: {final_chunk_index}. Returning this chunk.")
                 return [chosen_chunk_data]
            else:
                 logging.warning(f"LLM chose Chunk Index {final_chunk_index}, but it wasn't in the top {actual_top_n} semantic results!? Falling back to top semantic.")
                 return [semantic_results_data[0]] if semantic_results_data else [] # Ensure fallback isn't empty
        else:
            logging.info("LLM did not select a specific chunk or parsing failed. Falling back to top semantic result.")
            return [semantic_results_data[0]] if semantic_results_data else [] # Ensure fallback isn't empty

    except Exception as e:
        logging.error(f"Error calling LLM API or processing response: {e}", exc_info=True)
        logging.info("Falling back to top semantic match due to LLM error.")
        return [semantic_results_data[0]] if semantic_results_data else []

# --- Remove Standalone Execution --- 
# (Commented out the old processing loop and final output)
# # --- Store Results ---
# all_reasoning = []
# query_results = {}
#
# # --- Process Each Query ---
# print("Processing Queries...")
# queries = [... ] # Old hardcoded queries
# for i, query in enumerate(queries):
#     print(f"\n-- Query {i+1}/{len(queries)}: {query} --")
#     # ... old logic ...
#
# # --- Final Output ---
# print("\n--- Final Results (Top Semantic Match) ---")
# for q in queries:
#     # ... old output logic ...