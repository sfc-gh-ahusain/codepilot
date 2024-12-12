import configparser
import json
import logging
import os
import sys
import threading
import time
import warnings

from logging.handlers import RotatingFileHandler
from sentence_transformers import SentenceTransformer, models

MB = 1024*1024

class Config:

    def __init__(self):
        # file_pattern = [".c", ".cpp", ".py", ".java", ".txt", ".toml", ".sh"]
        self.file_pattern = None
        self.exclude_pattern = []

        # llm model path
        self.llm_model = None
        self.llm_model_path = None
        self.gpt4all_llama_path = None
        self.gpt4all_falcon_path= None
        self.llm_model_path_dict = {}

        # faiss setup
        self.embedding_model = None

        # CodePilot file setup
        self.metadata_dir = None
        self.model_dir = None
        self.checksum_file = None
        self.file_mapping_file = None
        self.faiss_index_file = None
        self.cost_estimates_file = None

        # Transformers setup
        self.transformer = None
        self.dimension_dict = {}

        self.logger = None


    def load_embedding_model(self):
        if self.transformer == "microsoft/codebert-base":
            # Manual setup for CodeBERT with mean pooling
            word_embedding_model = models.Transformer(self.transformer)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
            embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            # Direct loading for compatible models like MiniLM
            embedding_model = SentenceTransformer(self.transformer)

        return embedding_model

    def load_init(self, init_file_path=None):
        if not os.path.exists(init_file_path):
            print(f"Error: The configuration file '{init_file_path}' was not found.")
        else:
            try:
                conf = configparser.ConfigParser()
                conf.read(init_file_path)
            except configparser.ParsingError as e:
                print(f"""
                      Error: There was a problem parsing the config file '{init_file_path}': {e}
                    """)
            except configparser.MissingSectionHeaderError as e:
                print(f"""
                      Error: The config file '{init_file_path}' is missing a section header: {e}
                    """)

        self.logger = self.create_setup_logger(conf)
        self.start_log_repear()

        file_pattern_json = conf["pattern"]["file_pattern"]
        self.file_pattern = json.loads(file_pattern_json) 
        self.transformer = conf["model"].get("transformer", "all-MiniLM-L6-v2").strip().strip("'").strip('"').lower()
        self.embedding_model = self.load_embedding_model() # SentenceTransformer(Config.transformer)
        self.dimension_dict = {key.lower(): int(value) for key, value in conf["model_dimensions"].items()}

        self.llm_model = conf["model"].get("llm_model", "llama3")
        if self.llm_model == "llama3":
            self.llm_model_path = conf["llama3"].get("path", "").strip('"')
        elif self.llm_model == "gpt4falcon":
            self.llm_model_path = conf["gpt4falcon"].get("path", "").strip('"')
        else:
            self.logger.warning(f"Model unsupported {self.llm_model}, use llama3")
            self.llm_model = "llama3"
            self.llm_model_path = conf["llama3"].get("path", "").strip('"')
        
        self.llm_model_path = os.path.expanduser(self.llm_model_path)

        self.setup_metadata_file_paths(conf)

    def create_setup_logger(self, conf):
        # Create the directory if it doesn't exist
        os.makedirs(conf["paths"]["logs_dir"], exist_ok=True) 

        logger = logging.getLogger(conf["logging"]["log_file"])
        if conf["logging"]["level"] == "DEBUG":
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Create a rotating file handler
        log_file_path = conf["paths"]["logs_dir"].strip('"') 
        log_file_path = os.path.expanduser(log_file_path) + '/' + conf["logging"]["log_file"]
        max_bytes = int(conf["log_settings"].get("max_log_size", 1024*1024))
        backup_count = int(conf["log_settings"].get("backup_count", 5))
        handler = RotatingFileHandler(
            log_file_path,      
            maxBytes=max_bytes,  
            backupCount=backup_count
        )
        handler.setLevel(logging.DEBUG)
        # Set the format for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(handler)

        return logger


        # Periodic flush function
    def periodic_flush(self, interval=60):
        """Periodically flush the log handler."""
        while True:
            time.sleep(interval)
            for handler in self.logger.handlers:
                handler.flush()  # Force flush to disk


    def start_log_repear(self):
        # Start a background thread to periodically flush the logs
        flush_thread = threading.Thread(target=self.periodic_flush, args=(10,), daemon=True)
        flush_thread.start()


    # Function to flush logs on exit
    def flush_logs(self):
        """Flush and close the logger handlers."""
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        print("Logger flushed and closed.")


    def is_debug_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)


    def setup_metadata_file_paths(self, conf):
        self.metadata_dir = conf["paths"].get("metadata_dir", "./").strip('"')
        self.metadata_dir = os.path.expanduser(self.metadata_dir)
        self.model_dir = self.metadata_dir + '/' + self.llm_model
        self.checksum_file = self.model_dir + '/' + conf["filenames"].get("checksum_file_name", "checksums.json")
        self.file_mapping_file = self.model_dir + '/' + conf["filenames"].get("file_mapping_file_name", "file_mapping.json") 
        self.faiss_index_file = self.model_dir + '/' + conf["filenames"].get("faiss_index_file_name", "faiss_index.bin")
        self.cost_estimates_file = self.model_dir + '/' + conf["filenames"].get("cost_estimate_file_name", "cost_estimates.json")

        self.logger.info(f"""
                     Setup metadata dir path 
                     checksum {self.checksum_file} 
                     file_mapping {self.file_mapping_file} 
                     faiss_index {self.faiss_index_file} 
                     cost_estimate {self.cost_estimates_file}
                     """)

