import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import pdfplumber
import re
import glob

# --- Configuration ---
# Process only the specified Harry Potter book
input_files = [("Book_Collection/Harry Potter/Harry Potter and the Order of the Phoenix.pdf", "Harry Potter")]
CHUNKS_JSON_DIR = "chunks_json"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50
ENCODING_NAME = "cl100k_base"
# --- End Configuration ---

def extract_chapter_id(filename):
    """Extracts a generic document ID from filename (e.g., base name without extension)."""
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    return name

def generate_output_filename(input_filename, series_name):
    """Generates the output JSON filename based on the input PDF filename and series name."""
    basename = os.path.basename(input_filename)
    name, _ = os.path.splitext(basename)
    # Lowercase, underscores for spaces
    clean_name = name.lower().replace(" ", "_")
    output_dir = os.path.join(CHUNKS_JSON_DIR, series_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, f"{clean_name}.json"), clean_name

def read_and_extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            all_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
        text = '\n'.join(all_text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading/parsing PDF {file_path}: {e}")
        return None

def chunk_text_by_tokens(text, chunk_size, chunk_overlap, encoding_name="cl100k_base"):
    if not text:
        return [], None
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Error getting tokenizer '{encoding_name}': {e}. Falling back...")
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e2:
            print(f"Error getting fallback tokenizer: {e2}")
            return [], None

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    # Filter out any potentially empty chunks that might result from splitting
    chunks = [chunk for chunk in chunks if chunk.strip()]
    return chunks, tokenizer

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    processed_files = 0
    for input_file_path, series_name in input_files:
        output_chunks = []  # Reset list for each book
        # Check if input file exists
        if not os.path.isfile(input_file_path):
            print(f"--- Skipping: Input file not found - {input_file_path} ---")
            continue # Skip to the next file
        print(f"\n--- Processing: {input_file_path} ---")
        output_json_file, book_name = generate_output_filename(input_file_path, series_name)
        print(f"Output will be saved to: {output_json_file}")
        chapter_id = extract_chapter_id(input_file_path)
        print(f"Extracted Document ID: {chapter_id}")
        book_text = read_and_extract_text_from_pdf(input_file_path)
        if not book_text:
            print(f"Could not read or extract text from {input_file_path}. Skipping.")
            continue # Skip to the next file
        words = book_text.split()
        total_words = len(words)
        print(f"Total word count in book: {total_words}")
        chunks, _ = chunk_text_by_tokens(
            text=book_text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            encoding_name=ENCODING_NAME
        )
        if chunks:
            print(f"Generated {len(chunks)} chunks for the book.")
            num_chunks = len(chunks)
            for i, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if chunk_text:
                    chunk_data = {
                        "book_name": book_name,
                        "chunk_index": i,
                        "text": chunk_text
                    }
                    output_chunks.append(chunk_data)
                else:
                    print(f"Warning: Empty chunk generated at index {i} for {input_file_path}, skipping.")
        else:
            print(f"No chunks were generated for the book.")
        # Save the chunks for the current book
        if output_chunks:
            print(f"\nSaving {len(output_chunks)} chunks to {output_json_file}...")
            try:
                with open(output_json_file, 'w', encoding='utf-8') as f:
                    json.dump(output_chunks, f, indent=4, ensure_ascii=False)
                print(f"Successfully saved chunks to {output_json_file}.")
                processed_files += 1
            except Exception as e:
                print(f"Error saving chunks to JSON for {input_file_path}: {e}")
        else:
            print(f"\nNo valid chunks were generated or processed for {input_file_path}.")
    print(f"\n--- Processing Complete. Processed {processed_files} out of {len(input_files)} files. ---")


if __name__ == "__main__":
    main()