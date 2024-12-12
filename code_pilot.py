import atexit
import argparse
import asyncio

import code_pilot_cli
import cost_estimator
import faiss_utils
import llm_utils
import code_pilot_utils

from config import Config


class CodePilot:

    def __init__(self, config):
        self.faiss_index = None
        self.index_file_mapping = {}
        self.llm_model = None
        self.cost_estimates = None
        self.config = config


    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Snowflake code doc generator")
        parser.add_argument("repo_name", type=str, help="Repo name")
        parser.add_argument("repo_path", type=str, help="Repo URL")
        parser.add_argument("init_path", type=str, help = "initialization file path")
        args = parser.parse_args()

        return args


    def display_chunks(self, file_chunks):
        if not self.config.is_debug_enabled():
            # verbose logging disabled, do nothing
            return

        # Display chunked results for verification
        for file, chunks in file_chunks.items():
            self.config.logger.debug("\nFile: %s - Total Chunks: %ld", file, len(chunks))
            for i, chunk in enumerate(chunks):
                self.config.logger.debug("Chunk %ld:\n%s\n", i+1, chunk)


    def display_insights(self, file_insights):
        if not self.config.is_debug_enabled():
            # verbose logging disabled, do nothing
            return

        for file, insights in file_insights.items():
            self.config.logger.debug("\nInsights for File: %s - Total Insights: %ld", file, len(insights))
            for i, insight in enumerate(insights):
                self.config.logger.debug("Insight %ld:\n%s\n", i+1, insight)    


    def interactive_query_session(self):
        """
        Allows the user to ask multiple questions in a loop.

        Parameters:
        - faiss_index: Loaded FAISS index with embeddings.
        - index_file_mapping: Dictionary mapping FAISS index positions to insights.
        """
        while True:

            user_query = input("\n\nAsk your question (type 'exit' or 'quit' to stop): ").strip()

            if user_query.lower() in ['exit', 'quit']:
                print("Exiting the session.")
                break

            if user_query.lower() in ['exit', 'quit']:
                print("Ending the session.")
                break

            # Step 1: Process the query to extract entities and keywords
            entities, keywords = faiss_utils.process_query(user_query)
            self.config.logger.debug(f"Extracted entities: {entities}, keywords: {keywords}")
            user_query_embedding = faiss_utils.get_query_embedding(self.config, user_query)

            distances, indices = faiss_utils.query_faiss_index(self.faiss_index, user_query_embedding, 5)

            # Step 3: Fetch results and prepare the context
            insights, index_files = code_pilot_utils.fetch_insights(self.config, indices, self.index_file_mapping)
            if insights:
                # Step 4: Estimate cost for running the query, let user choose to proceed
                estimated_cost = cost_estimator.estimate_query_cost(self.config, index_files, self.cost_estimates) 

                print(f"""
                      Estimated Cost (excludes response tokens cost): ${estimated_cost:.6f}.
                      """)
                user_confirmation = input("Do you want to proceed with this cost? (yes/no): ").strip().lower()
                if user_confirmation != 'yes':
                    print("Query canceled.")
                    continue  # Skip to the next query if user doesn't confirm

                # Step 5: Construct the prompt dynamically using the user's question and the insights
                input_prompt = code_pilot_utils.construct_prompt(insights, user_query, entities, keywords)

                # Step 6: Consult with LLM and generate response
                if input_prompt:
                    response = llm_utils.generate_response(self.config, input_prompt, self.llm_model)
                    actual_cost = cost_estimator.calculate_actual_cost(self.config.embedding_model, estimated_cost, response)
                    if response:
                        print(f"""Actual cost: {actual_cost:.6f}""")
                        print(f"Response: {response}")
            else:
                print("No relevant insights found.")


    def clone_update_repo(self, repo_name, repo_path, file_mapping_file):
        code_pilot_utils.clone_or_update_repo(self.config, repo_path, repo_name)

        # Load existing file mapping
        file_mapping = code_pilot_utils.load_file_mapping(self.config, file_mapping_file)
        existing_files = set(file_mapping.keys())
        self.config.logger.info(f"loaded {file_mapping.keys()} existing {existing_files}")

        return file_mapping, existing_files


    def list_refresh_repo_files(self, repo_name, existing_files, file_mapping):
        current_files = code_pilot_utils.list_code_files(self.config, repo_name)

        # Check for deleted files
        for file_path in existing_files:
            if file_path not in current_files:
                self.config.logger.info(f"File {file_path} has been removed. Updating mappings.")
                del file_mapping[file_path]  # Remove from mapping

        # Load previous checksums to track changes
        previous_checksums = code_pilot_utils.load_previous_checksums(self.config.checksum_file)
        # Index new/updated files
        return code_pilot_utils.get_changed_files(self.config, current_files, previous_checksums)


    def generate_file_insights(self, file_chunks):
        file_insights = {}
        for filename, chunks in file_chunks.items():
            insights = []
            for chunk_data in chunks:
                chunk_embedding = chunk_data["embedding"]

                # Process the chunk's text with the LLM
                try:
                    # Create a dictionary for the single chunk to pass to the LLM
                    chunk_input = {filename: [chunk_data]}  # Wrap the single chunk in a list
                    insight = llm_utils.process_chunks_with_llm(self.config, chunk_input, self.llm_model)
                    insights.append({"insight": insight[filename][0]["insight"], "embedding": chunk_embedding})  # Extract insight
                except Exception as e:
                    self.config.logger.error(f"LLM processing failed for chunk in file {filename}: {e}")

            file_insights[filename] = insights
            cost_estimator.update_cost_for_file(self.config, filename, chunks, self.cost_estimates)

        self.display_insights(file_insights)
        return file_insights


    def load_faiss_index(self, file_insights):
        # Leverage vector database to store the insights
        dimension = 0
        if not file_insights:
            self.config.logger.warning("No files processed. Skipping FAISS index creation.")
            dimension = self.config.dimension_dict[self.config.transformer]
        else:    
            dimension = len(next(iter(file_insights.values()))[0]["embedding"])
            self.config.logger.info(f"Transformer model dimension {dimension}")

        return faiss_utils.load_faiss_index(self.config, dimension)


    def loadResources(self):
        args = self.parse_arguments()

        self.config.load_init(args.init_path)
        self.config.logger.info("Welcome to CodePilot!")

        self.llm_model = llm_utils.load_gpt4all_model(self.config.llm_model_path)
        self.config.logger.info(f"LLM model selected {self.config.llm_model}")

        # Clone/update repo
        file_mapping, existing_files = self.clone_update_repo(
            args.repo_name, args.repo_path, self.config.file_mapping_file)
        self.cost_estimates = cost_estimator.load_cost_estimates(self.config)

        changed_files = self.list_refresh_repo_files(args.repo_name, existing_files, file_mapping)

        # Process the updated file chunks
        file_chunks = code_pilot_utils.process_files_for_chunking(config, changed_files)
        self.display_chunks(file_chunks)

        # Parse and understand each chunk using the LLM, adding insights to the `file_insights`
        file_insights = self.generate_file_insights(file_chunks)
        cost_estimator.save_cost_estimates(self.config, self.cost_estimates)

        self.faiss_index = self.load_faiss_index(file_insights)

        # Prepare embeddings and file mapping for FAISS
        embeddings = []
        index = 0

        for filename in file_mapping.keys():
            self.index_file_mapping[index] = {"filename" : filename, "insight" : file_mapping[filename]}
            index += 1

        for filename, insights in file_insights.items():
            for insight_data in insights:
                embeddings.append(insight_data["embedding"])
                file_mapping[filename] = insight_data["insight"]
                self.index_file_mapping[index] = {"filename": filename, "insight": insight_data["insight"]}
                index += 1

        code_pilot_utils.save_file_mapping(self.config, file_mapping)

        # Persist the computed index
        if not embeddings:
            self.config.logger.warning("No embeddings found. Skipping addition to FAISS index.")
        else:
            try:
                # Add embeddings to FAISS index
                faiss_utils.add_embeddings_to_faiss(self.config, self.faiss_index, embeddings)
                faiss_utils.persist_index(self.config, self.faiss_index)
            except Exception as e:
                self.config.logger.error(f"An error occurred while adding embeddings to the FAISS index.")
                self.config.logger.error(f"Error details: {str(e)}")


if __name__ == "__main__":
    config = Config()
    # Register the flush function to be called on exit
    atexit.register(config.flush_logs)

    codepilot = CodePilot(config)
    codepilot.loadResources()
    # code.interactive_query_session()
    cli = code_pilot_cli.CodePilotCLI(config, codepilot)
    asyncio.run(cli.run())

