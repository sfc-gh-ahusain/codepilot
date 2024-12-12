import re

from config import Config
from gpt4all import GPT4All


# Initialize the model
def load_gpt4all_model(model_path=None):
    model = GPT4All(model_path)

    return model


def ask_llm(config, text, llm_model):
    try:
        # Process the chunk with the LLM model and get a summary or insight
        response = llm_model.generate(text)
        return response
    except Exception as e:
        config.logger.error("Error in understanding chunk: %s", e)
        return None


def process_chunks_with_llm(config, file_chunks, llm_model):
    file_insights = {}

    for filename, chunks in file_chunks.items():
        insights = []

        for chunk_data in chunks:
            chunk_text = chunk_data["text"]

            # Process the chunk's text with the LLM
            try:
                insight = ask_llm(config, chunk_text, llm_model)
                insights.append({"insight": insight, "embedding": chunk_data["embedding"]})
            except Exception as e:
                config.logger.error(f"LLM processing failed for chunk in file {filename}: {e}")
                insights.append({"insight": None, "embedding": chunk_data["embedding"]})

        # Store insights per file
        file_insights[filename] = insights

    return file_insights


def generate_response(config, input_prompt, llm_model):
    try:
        return ask_llm(config, input_prompt, llm_model)
    except Exception as e:
        config.logger.error(f"Error processing query: {e}")
        return None

