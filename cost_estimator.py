import json

from config import Config

def load_cost_estimates(config):
    """
    Load cost estimates from a JSON file.

    Args:
        filepath (str): The path to the cost estimates file.

    Returns:
        dict: A dictionary containing cost estimates for files.
    """
    try:
        with open(config.cost_estimates_file, 'r') as file:
            cost_estimates = json.load(file)
            return cost_estimates
    except FileNotFoundError:
        config.logger.warning(f"""
                              No cost estimates file found at {config.cost_estimates_file}. Starting fresh.
                              """)
        return {}  # Return an empty dictionary if the file doesn't exist
    except json.JSONDecodeError:
        config.logger.error(f"""
                            Error decoding JSON from {config.cost_estimates_file}. Returning an empty dictionary.
                            """)
        return {}  # Return an empty dictionary if there's a JSON error


def save_cost_estimates(config, cost_estimates):
    config.logger.info(f"Saving cost_estmates to {config.cost_estimates_file}")

    with open(config.cost_estimates_file, 'w') as file:
        json.dump(cost_estimates, file, indent=2)


def estimate_cost(embedding_model, chunk, response_length=0, cost_per_token=0.00001):
    # Tokenize the input chunk
    input_tokens = embedding_model.tokenizer.tokenize(chunk)
    input_token_count = len(input_tokens)
    
    # Total token count includes both input and estimated response tokens
    total_token_count = input_token_count + response_length
    cost_estimate = round(total_token_count * cost_per_token, 6)

    return input_token_count, cost_estimate


def calculate_actual_cost(embedding_model, estimated_cost, response, cost_per_token=0.00001):
    # Tokenize response for actual cost calculation
    response_tokens = embedding_model.tokenizer.tokenize(response)
    response_length = len(response_tokens)
    
    # Re-estimate with actual response length
    actual_cost = estimated_cost + round(response_length * cost_per_token, 6)
    
    return actual_cost


def update_cost_for_file(config, file_path, chunks, cost_estimates):
    tokens = 0
    cost_estimate = 0
    for chunk_data in chunks:
        chunk_text = chunk_data["text"]
        t, c = estimate_cost(config.embedding_model, chunk_text)
        tokens += t
        cost_estimate += c

    config.logger.info(f"""
                        Estimated cost 
                        file: {file_path} 
                        tokens: {tokens} 
                        cost: {cost_estimate}
                        """)
    cost_estimates[file_path] = {"tokens": tokens, "cost": cost_estimate}


def estimate_query_cost(config, index_files, cost_estimates):
	total_cost = 0
	file_used = 0
	token_count = 0

	if index_files:
		for file_path in index_files:
			if file_path in cost_estimates:
				total_cost += cost_estimates[file_path]['cost']
				file_used += 1
				token_count += cost_estimates[file_path]['tokens']
	else:
		config.logger.warning(f"No cost estimate found for {file_path}. Estimating cost...")

	config.logger.info(f"""
                          Estimated total cost for the query 
                          files: {file_used} 
                          tokens: {token_count} 
                          cost: {total_cost}
                        """)

	return total_cost
