# Soumita_Deb

A comprehensive project repository showcasing advanced machine learning and RAG (Retrieval-Augmented Generation) pipeline implementations.

## 📋 Project Overview

This repository contains implementations of modern AI and machine learning techniques, with a focus on:

- **Retrieval-Augmented Generation (RAG)**: Advanced RAG pipelines combining semantic and keyword-based retrieval
- **Hybrid Search**: Integration of dense vector retrieval with sparse BM25 retrieval
- **Query Decomposition**: Breaking down complex queries into manageable sub-queries
- **Local LLM Integration**: Using local GGUF models for cost-effective inference
- **Embedding Models**: HuggingFace embeddings for semantic similarity search

## 🚀 Key Features

### Advanced RAG Pipeline
- Combines multiple retrieval strategies for improved accuracy
- Uses reciprocal rank fusion to merge results from different retrievers
- Supports local LLM execution with LlamaCPP
- Implements query decomposition for better recall

### Hybrid Retrieval System
- **Dense Retrieval**: Semantic similarity search using HuggingFace embeddings
- **Sparse Retrieval**: BM25-based keyword matching
- **Result Fusion**: Intelligent combination of results using reciprocal rank fusion

### Local Model Support
- Download and run models from HuggingFace Hub
- Support for GGUF format models (quantized for efficiency)
- CPU and GPU acceleration support

## 📦 Dependencies

```
llama-index
llama-index-llms-openai
llama-index-embeddings-openai
llama-index-embeddings-huggingface
llama-index-llms-llama-cpp
llama-index-retrievers-bm25
llama-index-readers-file
sentence-transformers
nest-asyncio
torch
huggingface-hub
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Soumita2026/Soumita_Deb.git
cd Soumita_Deb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support with llama-cpp-python:
```bash
pip install llama-cpp-python --force-reinstall --no-deps
```

## 📚 Usage

### Setting Up the Knowledge Base

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

# Load documents from directory
documents = SimpleDirectoryReader("path/to/documents").load_data()

# Create vector index
vector_index = VectorStoreIndex.from_documents(documents)
```

### Creating a Hybrid Retriever

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Initialize retrievers
vector_retriever = vector_index.as_retriever(similarity_top_k=4)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)

# Create fusion retriever
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=6,
    num_queries=4,
    mode="reciprocal_rerank",
)
```

### Running Queries

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(
    retriever=fusion_retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query("Your question here")
print(response)
```

## 🤖 Using Local LLMs

Download and use local models with LlamaCPP:

```python
from llama_index.llms.llama_cpp import LlamaCPP
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF",
    filename="Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf"
)

# Initialize LLM
llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    model_kwargs={"n_gpu_layers": -1},  # Use GPU if available
)
```

## 🧠 Embedding Models

This project uses HuggingFace embeddings for semantic search:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda"  # or "cpu"
)

Settings.embed_model = embed_model
```

## 📊 Configuration

Key configuration options in the pipeline:

- **chunk_size**: Document chunk size (default: 256 tokens)
- **chunk_overlap**: Overlap between chunks (default: 40 tokens)
- **similarity_top_k**: Number of top results to retrieve (default: 4-6)
- **num_queries**: Number of sub-queries for decomposition (default: 4)
- **temperature**: LLM temperature for generation (default: 0.1)

## 🔍 Advanced Features

### Query Decomposition
The system automatically breaks down complex queries into simpler sub-queries to improve retrieval coverage.

### Reciprocal Rank Fusion
Results from multiple retrievers are combined using reciprocal rank fusion, which:
- Assigns scores based on rank positions
- Combines scores from different retrievers
- Re-ranks results for optimal relevance

### Citation Support
Retrieved documents are tracked and can be used to provide citations in responses, improving transparency and trust.

## 📝 Example Workflow

```python
# 1. Load and prepare documents
documents = SimpleDirectoryReader("knowledge_base").load_data()

# 2. Create indexes
vector_index = VectorStoreIndex.from_documents(documents)

# 3. Set up retrievers
vector_retriever = vector_index.as_retriever(similarity_top_k=4)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)

# 4. Create fusion retriever
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=6,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=True,
)

# 5. Set up response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="compact")

# 6. Create query engine
query_engine = RetrieverQueryEngine(
    retriever=fusion_retriever,
    response_synthesizer=response_synthesizer,
)

# 7. Run queries
response = query_engine.query("What is RAG and why is it important?")
print(response)
```

## 🛠️ Troubleshooting

### OpenAI API Quota Exceeded
If using OpenAI models, ensure your API key has sufficient quota or switch to local models.

### Out of Memory
For local models, reduce `context_window` or use quantized models (Q4_K_M format).

### Slow Retrieval
Optimize by:
- Reducing `similarity_top_k`
- Using fewer retrievers
- Enabling GPU for embeddings

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Soumita2026**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📮 Contact

For questions or feedback, please open an issue on the repository.

---

**Last Updated**: June 2026

For more information on RAG systems and LlamaIndex, visit:
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [HuggingFace Models](https://huggingface.co/models)
- [LlamaCPP](https://github.com/ggerganov/llama.cpp)
