import os
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import ContextChatEngine


def get_chat_engine(api_key: str):
    """Initializes the RAG system and returns a clean conversation engine."""
    # Configure global settings
    os.environ["OPENAI_API_KEY"] = api_key
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=40)

    # Verify or generate the data folder
    os.makedirs("tech_kb", exist_ok=True)

    # Read the text files from your knowledge base directory
    if not os.listdir("tech_kb"):
        # Quick fallback if folder is completely empty
        with open("tech_kb/sample.txt", "w", encoding="utf-8") as f:
            f.write("Retrieval-Augmented Generation (RAG) mitigates LLM hallucinations using relevant context.")

    documents = SimpleDirectoryReader("tech_kb").load_data()
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)

    # ContextChatEngine handles full conversation history natively
    return ContextChatEngine.from_defaults(
        retriever=vector_retriever,
        llm=Settings.llm,
        system_prompt="You are a precise technical assistant. Answer strictly using the provided context."
    )