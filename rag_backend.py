import os
import requests
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.chat_engine import ContextChatEngine


class DirectOllamaLLM(CustomLLM):
    """A direct connector that talks to your desktop Ollama app over local network ports."""
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "llama3.2:3b"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Talks straight to the active app service in your system taskbar
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload).json()
            return CompletionResponse(text=response.get("response", ""))
        except Exception:
            return CompletionResponse(text="Error: Could not reach Ollama. Is the app open in your taskbar?")

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming is disabled for this assignment interface layout.")


def get_chat_engine():
    """Initializes RAG securely without loading any unstable external dependencies."""
    # Register our direct internal layout helper
    Settings.llm = DirectOllamaLLM()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=40)

    os.makedirs("tech_kb", exist_ok=True)
    if not os.listdir("tech_kb"):
        with open("tech_kb/sample.txt", "w", encoding="utf-8") as f:
            f.write("Retrieval-Augmented Generation (RAG) mitigates LLM hallucinations using relevant context.")

    documents = SimpleDirectoryReader("tech_kb").load_data()
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)

    return ContextChatEngine.from_defaults(
        retriever=vector_retriever,
        llm=Settings.llm,
        system_prompt="You are a precise technical assistant. Answer strictly using the provided context."
    )