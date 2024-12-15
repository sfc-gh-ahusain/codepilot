# CODEPILOT: Contextual Code & Document Exploration Tool

CODEPILOT is a powerful semantic search and document analysis tool designed to streamline the exploration of codebases and textual repositories. By leveraging state-of-the-art machine learning models and FAISS as a vector database, CODEPILOT enables users to perform context-aware searches, gain actionable insights, and answer complex queries directly from documents. 

CODEPILOT follow per-query billing model; it generates estimated cost of running query before processing any query, enables user to choose if they are willing to proceed. 

---

## Key Features

### 1. **Semantic Search**
CODEPILOT uses embeddings to enable advanced semantic search capabilities. Instead of simple keyword matching, it understands the meaning behind user queries, providing more accurate and relevant results.

### 2. **Actionable Insights**
CODEPILOT extracts structured insights from textual data, such as entities, keywords, and context, to provide a deeper understanding of the content.

### 3. **Git Repository Support**
Designed with developers in mind, CODEPILOT seamlessly integrates with Git repositories, indexing code and documentation for easy exploration.

### 4. **Efficient Chunking and Embedding**
CODEPILOT divides large documents into manageable chunks, generating embeddings for each chunk. This enables scalable indexing and retrieval of data.

### 5. **Interactive Query Interface**
CODEPILOT offers an intuitive query interface that supports iterative and conversational searches, helping users refine their queries to get precise answers.

---

## How It Works

### Insights Generation
- **Extract Keywords and Entities**: Automatically identifies keywords and named entities within the text to enhance search accuracy.
- **Contextual Analysis**: Generates insights to understand user intent and provide meaningful responses.

### Embedding with FAISS
- **Vector Database**: Uses FAISS to store and retrieve document embeddings efficiently.
- **Similarity Search**: Finds relevant chunks based on semantic similarity between query embeddings and document embeddings.

### Indexing and Chunking
- **Dynamic Chunking**: Splits documents into token-limited chunks to optimize embedding generation.
- **Streaming Support**: Reads and processes large files incrementally for better memory management.

---

## Installation

### Requirements
- Python 3.8+
- Libraries: PyTorch, FAISS, SentenceTransformers, and FastAPI

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/codepilot.git
   cd codepilot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   python main.py
   ```

---

## Usage

1. **Index Files**: Point CODEPILOT to a directory or Git repository to index files.
2. **Ask Questions**: Use the interactive query interface to explore the indexed content.
3. **Iterate and Refine**: Leverage conversational capabilities to refine queries and improve search outcomes.

---

## Contributions

We welcome contributions! Please open an issue or submit a pull request to help us improve CODEPILOT.

---

## License

This project is licensed under the [Apache 2.0](LICENSE.md). It is free to use and modify for personal and academic purposes but cannot be used for commercial purposes without explicit permission.

---

## Contact

For questions, feedback, or feature requests, reach out via email at [atahusain.b@gmail.com] or open an issue on GitHub.

