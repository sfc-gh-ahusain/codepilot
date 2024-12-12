import faiss
import os
import numpy as np

import code_pilot_utils

from config import Config
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
from transformers import pipeline

def load_faiss_index(config, dimension):
    if os.path.exists(config.faiss_index_file):
        config.logger.info(f"""
                           Loading FAISS index from 
                           file '{config.faiss_index_file}' 
                           dimension {dimension} 
                           checksum {code_pilot_utils.calculate_checksum(config.faiss_index_file)}...
                        """)
        faiss_index = faiss.read_index(config.faiss_index_file)
        if faiss_index.ntotal == 0:
            config.logger.warning("Loaded FAISS index is empty.")
        return faiss_index
    else:
        config.logger.info(f"""
                           No index file '{config.faiss_index_file}' found, creating with dimension '{dimension}' ...
                           """)
        return faiss.IndexFlatL2(dimension) 


def flatten_insights(insights):
    # Create lists to hold summaries and corresponding file names
    summary_list = []
    file_mapping = []

    for file, summaries in insights.items():
        for summary in summaries:
            summary_list.append(summary)
            file_mapping.append(file)  # Keep track of the corresponding file

    return summary_list, file_mapping


def get_query_embedding(config, query):
    return config.embedding_model.encode([query]).astype('float32')


def query_faiss_index(index, query_embedding, top_k=5):
    # Convert the query to an embedding
    distances, indices = index.search(query_embedding, top_k)

    return distances, indices


def add_embeddings_to_faiss(config, faiss_index, embeddings):
    """
    Adds a list of embeddings to the FAISS index.

    Parameters:
    - faiss_index: FAISS index object
    - embeddings: List of embeddings (each embedding is a list or numpy array)

    Returns:
    - faiss_index: The FAISS index with added embeddings
    """
    # Convert the embeddings list to a numpy array if it isn't one already
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype='float32')

    # Ensure embeddings are 2D for FAISS (batch insertion)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    try:
        # Add embeddings to FAISS index
        faiss_index.add(embeddings)
        config.logger.info(f"Successfully added {len(embeddings)} embeddings to FAISS index.")
    except Exception as e:
        print(f"Failed to add embeddings to FAISS index: {e}")
        raise

    return faiss_index


def encode_large_text_with_embedding_model(embedding_model, text, max_length=512):
    # Tokenize the text and determine if chunking is needed
    tokens = embedding_model.tokenizer.tokenize(text)
    num_tokens = len(tokens)

    if num_tokens <= max_length:
        # Encode directly if text length is within the limit
        return embedding_model.encode(text).tolist()

    # If the text is too long, split into smaller chunks
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        # If we reach max length, finalize the current chunk
        if current_length >= max_length:
            chunks.append(embedding_model.tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0

    # Handle any remaining tokens in the last chunk
    if current_chunk:
        chunks.append(embedding_model.tokenizer.convert_tokens_to_string(current_chunk))

    # Encode each chunk and aggregate
    chunk_embeddings = [embedding_model.encode(chunk) for chunk in chunks]
    final_embedding = np.mean(chunk_embeddings, axis=0)  # Mean pooling over chunks

    return final_embedding.tolist()

def embed_text(embedding_model, text):
    """
    Embeds the given text using a pre-trained embedding model.

    Parameters:
    - text: The input text to be embedded.

    Returns:
    - The embedding of the input text.
    """
    return embedding_model.encode(text).tolist()


def embed_text_batch(embedding_model, texts):
    """
    Embeds a batch of texts using a pre-trained embedding model.

    Parameters:
    - texts: A list of input texts to be embedded.

    Returns:
    - A list of embeddings for each text in the input batch.
    """
    embeddings = embedding_model.encode(texts) 
    return embeddings.tolist()


def persist_index(config, faiss_index):
    faiss.write_index(faiss_index, config.faiss_index_file)
    config.logger.info(f"""
                       Persist FAISS index 
                       filepath {config.faiss_index_file} 
                       checksum {code_pilot_utils.calculate_checksum(config.faiss_index_file)}
                    """)


# Load the sentiment analysis model
# nlp_pipeline = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english", device=-1)
nlp_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=-1)


def process_query(user_query):
    """Process the user query to extract intents and entities."""
    entities = nlp_pipeline(user_query)
    keywords = extract_keywords(user_query)

    return entities, keywords


 #def extract_keywords(query):
 #   """Simple keyword extraction. This can be enhanced further."""
 #  return [word for word in query.split() if len(word) > 3]

def extract_keywords(query: str):
    """
    Extract keywords from a query mixing natural language and code snippets.
    """
    nlp = English()
    nlp.add_pipe('sentencizer')  # Tokenize sentences
    doc = nlp(query)

    # Natural language keywords
    text_keywords = [
        token.text for token in doc if len(token.text) > 3 and token.is_alpha
    ]

    # Extract code-like keywords (e.g., function_name(), class_name, variables)
    code_keywords = [
        token.text for token in doc if "(" in token.text or "_" in token.text
    ]

    return list(set(text_keywords + code_keywords))