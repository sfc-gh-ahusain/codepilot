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
