import chardet
import hashlib
import git
import json
import numpy as np
import re
import os

import faiss_utils

from config import Config

MAX_TOKEN_LENGTH = 256

def save_file_mapping(config, file_mapping):
    if not config.file_mapping_file:
        config.logger.debug(f"Skip mapping save..")
        return

    with open(config.file_mapping_file, 'w') as f:
        json.dump(file_mapping, f)


def load_file_mapping(config, filepath=None):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        config.logger.warning(f"FileMapping file {filepath} doesn't exists")
        return {}  # Return empty dict if the file doesn't exist


def load_previous_checksums(checksum_file=None):
    # Load previous checksums from a JSON file
    if os.path.exists(checksum_file):
        with open(checksum_file, 'r') as f:
            return json.load(f)
    return {}


def save_checksums(checksums, checksum_file=None):
    if checksums is None:
        return

    with open(checksum_file, 'w') as f:
        json.dump(checksums, f)


def get_changed_files(config, code_files, previous_checksums):
    # Identify files that have changed based on checksums
    changed_files = []
    current_checksums = {}

    for file in code_files:
        checksum = calculate_checksum(file)
        current_checksums[file] = checksum
        if file not in previous_checksums or previous_checksums[file] != checksum:
            changed_files.append(file)

    # Save updated checksums for future runs
    save_checksums(current_checksums, config.checksum_file)
    return changed_files


def clone_or_update_repo(config, repo_url, local_path="repo"):
    # Clone the GitHub repository if it hasn't been cloned
    if not os.path.exists(local_path):
        git.Repo.clone_from(repo_url, local_path)
        config.logger.info(f"Repository cloned to {local_path}")
    else:
        # If the repo exists, pull the latest changes
        repo = git.Repo(local_path)
        repo.remotes.origin.pull()
        config.logger.info(f"Repository updated at {local_path}")


def list_code_files(config, local_path):
    """
    Recursively list code files in the repo, with options to include specific
    file extensions and exclude certain extensions.

    :param local_path: Path to the local directory
    :return: List of matching code files
    """

    # Recursively list code files in the repo
    code_files = []
    for root, _, files in os.walk(local_path):
        for file in files:
            # Check if file ends with specified extensions and does not end with any exclude extensions
            if (any(file.endswith(ext) for ext in config.file_pattern) and
                not any(file.endswith(ext) for ext in config.exclude_pattern)):
                code_files.append(os.path.join(root, file))
    return code_files


def calculate_checksum(file_path):
    # Calculate the checksum of a file
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def read_file_with_encoding(config, file_path):
    # Detect file encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        config.logger.debug(f"File {file_path} detected encoding: {encoding}")

    # Read the file with the detected encoding
    config.logger.debug(f"Open file {file_path}")
    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        content = file.read()

    return content


def read_file_streaming_with_encoding_and_chunking(config, file_path, embedding_model, max_tokens=MAX_TOKEN_LENGTH):
    """Streams the file, accumulates text, and chunks it when exceeding max_tokens."""
    
    try:
        # First, detect the encoding by reading a small portion of the file
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Read first 10,000 bytes to detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            config.logger.debug(f"File {file_path} detected encoding: {encoding}")
        
        # Initialize an empty string to accumulate text and a list for chunks
        text = ""
        chunks = []
        
        # Open the file for streaming reading with detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            for line in file:
                text += line  # Accumulate text from the current line
                if len(text) > max_tokens:  # If accumulated text exceeds max tokens
                    new_chunks = split_into_chunks(config, text, embedding_model, file_path)
                    chunks.extend(new_chunks)
                    text = ""  # Reset text buffer

            # Process any remaining text that didn't exceed max_tokens
            if text:
                new_chunks = split_into_chunks(config, text, embedding_model, file_path)
                chunks.extend(new_chunks)

        config.logger.info(f"Processed {len(chunks)} chunks from file {file_path}")
        return chunks

    except Exception as e:
        config.logger.error(f"Error processing file {file_path}: {e}")
        return []


def split_into_chunks(config, text, embedding_model, file_path, max_tokens=MAX_TOKEN_LENGTH, overlap=50):
    # Tokenize and split text into chunks within max token limit, with overlap
    tokens = embedding_model.tokenizer.encode(text, truncation=True)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunk_text = embedding_model.tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # Move window by (max_tokens - overlap)

    config.logger.info(f"Split into chunks with overlap {file_path} count {len(chunks)}")

    return chunks

""""
    # Tokenize and split text into chunks within max token limit
    tokens = embedding_model.tokenizer.encode(text, max_length=max_tokens, truncation=True)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = embedding_model.tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)

    config.logger.info(f"Split into chunks {file_path} count {len(chunks)}")

    return chunks
"""

def chunk_file(config, file_path, embedding_model):
    try:
        text = read_file_with_encoding(config, file_path)
        config.logger.debug(text)
    except Exception as e:
        config.logger.error(f"File parse error file {file_path}", exec_info=True)
    
    chunks = split_into_chunks(config, text, embedding_model, file_path)

    # Log each chunk's size for debugging
    if config.is_debug_enabled:
        for i, chunk in enumerate(chunks):
            config.logger.debug(f"Chunk {i+1}/{len(chunks)} length: {len(chunk)}")

    # Ensure all chunks have content
    assert all(len(chunk) > 0 for chunk in chunks), "One or more chunks are empty."

    return chunks

def generate_embeddings_in_batches(config, chunks, embedding_model, batch_size=10):
    chunk_data = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            # Create embeddings for the current batch of chunks
            batch_embeddings = faiss_utils.embed_text_batch(embedding_model, batch)

            for j, chunk in enumerate(batch):
                embedding = batch_embeddings[j]
                # Validate embedding dimension
                assert len(embedding) == config.dimension_dict[config.transformer], \
                    f"Embedding dimension mismatch for chunk {i + j}"
                
                chunk_data.append({"text": chunk, "embedding": embedding})
                config.logger.debug(f"Batch processed: Embedding for chunk {i + j} generated.")
        except Exception as e:
            config.logger.error(f"Embedding failed for batch {i // batch_size} in file: {e}")
            break  # Continue with the next batch if current batch fails

    return chunk_data

""""
def process_files_for_chunking(config, files_to_index):
    config.logger.info(f"Files to index {len(files_to_index)}")
    file_chunks = {}
    total_chunks = 0
    for file in files_to_index:
        # chunks = chunk_file(config, file, config.embedding_model)
        chunks = read_file_streaming_with_encoding_and_chunking(config, file, config.embedding_model)

        if not chunks:
            config.logger.error(f"No chunks created for file: {file}")
            continue

        # Process embeddings for the chunks in batches
        chunk_data = generate_embeddings_in_batches(config, chunks, config.embedding_model)

        # Ensure embeddings were created
        if len(chunk_data) != len(chunks):
            config.logger.error(f"Mismatch between chunks and embeddings for file: {file}")
            continue

        file_chunks[file] = chunk_data
        total_chunks += len(chunk_data)
        config.logger.info(f"Processed file {file} with {len(chunk_data)} chunks")

        # Clear intermediate variables to optimize memory
        del chunks
        del chunk_data

    config.logger.info(f"Total files processed: {len(files_to_index)}; Total chunks: {total_chunks}")
    return file_chunks
"""


def process_files_for_chunking(config, files_to_index):
    config.logger.info(f"Files to index {len(files_to_index)}")

    file_chunks = {}
    total_chunks = 0
    for file in files_to_index:
        chunks = chunk_file(config, file, config.embedding_model)

        if not chunks:
            config.logger.error(f"No chunks created for file: {file}")
            continue

        # Prepare each chunk with text and embeddings
        chunk_data = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = faiss_utils.embed_text(config.embedding_model, chunk)

                assert (
                    len(embedding) == config.dimension_dict[config.transformer]
                ), f"Embedding dimension mismatch for chunk {i} {len(embedding)} {config.dimension_dict}"

                chunk_data.append({"text": chunk, "embedding": embedding})
                config.logger.debug(f"Chunk {i+1} embedding generated.")
            except Exception as e:
                config.logger.error(f"Embedding failed for chunk {i} in file {file}: {e}")
        
        # Ensure all embeddings were created
        if len(chunk_data) != len(chunks):
            config.logger.error(f"Mismatch between chunks and embeddings for file: {file}")
            continue
        
        file_chunks[file] = chunk_data
        total_chunks += len(chunk_data)
        config.logger.info(f"Processed file {file} with {len(chunk_data)} chunks")
    
    config.logger.info(f"Total files processed: {len(files_to_index)}; Total chunks: {total_chunks}")
    return file_chunks


def fetch_insights(config, indices, index_file_mapping):
    insights = []
    index_files = []
       
    config.logger.debug(f"Retried indices {indices}")
    for idx in indices:
        # Ensure idx is an integer (if indices is an array, convert to int)
        if isinstance(idx, np.ndarray):
            idx = int(idx[0])  # or just use idx[0] if you expect a single index

        if 0 <= idx < len(index_file_mapping):
            file_info = index_file_mapping[idx]
            insights.append(file_info['insight'])
            index_files.append(file_info['filename'])
        else:
            config.logger.error(f"Index {idx} is out of range.")

    return insights, index_files


def construct_prompt(insights, user_query, entities, keywords):
    if not insights:
        return None

    # Safeguard for optional parameters
    entities = entities or []
    keywords = keywords or []

    # Construct dynamic elements
    context_text = "\n\n".join(insights)
    keywords_text = f"Keywords: {', '.join(keywords)}\n" if keywords else ""
    entities_text = f"Entities: {', '.join([entity.get('word', '') for entity in entities])}\n" if entities else ""

    # Build the input prompt
    input_prompt = (
        f"Based on the following insights:\n"
        f"{context_text}\n\n"
        f"{keywords_text}"
        f"{entities_text}"
        f"Answer the following question concisely and clearly:\n"
        f"{user_query}\n\n"
        f"Focus on providing a high-level explanation of the purpose and functionality without reiterating the query, "
        f"using numerical sequences, unnecessary symbols (e.g., '*', '~'), or technical jargon. "
        f"Do not include any raw code, such as variable names, method signatures, or other technical details. "
        f"Respond in plain language to make it easy to understand."
    )    

    return input_prompt

    """"

    # Construct the input prompt
    if insights:
        context = "\n\n".join(insights) if insights else ""
        input_prompt = (
            f"Based on the following insights and the user's intent:\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Entities: {', '.join([entity['word'] for entity in entities])}\n"
            f"{context}\n\n"
            f"Answer the question: {user_query}\n\n"
            f"Please provide a concise, clear, and focused explanation without including numerical sequences, unnecessary symbols like '*', '~', or technical jargon. "
            # f"Do not include any raw code, such as variable names, method signatures, or other technical details. "
            # f"Focus on describing the purpose and functionality of the class or method in simple terms."            
        )
        return input_prompt

    return None
    """
