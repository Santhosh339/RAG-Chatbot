import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore/faiss_index"


def build_vectorstore(chunks, api_key: str):
    """
    Create FAISS vector store from document chunks and save locally.

    Args:
        chunks: List of Document chunks
        api_key: OpenAI API key for embeddings
    """
    os.environ["OPENAI_API_KEY"] = api_key

    print("[Embeddings] Creating embeddings using OpenAI text-embedding-ada-002...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save locally
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"[Embeddings] Vectorstore saved to '{VECTORSTORE_PATH}'")

    return vectorstore


def load_vectorstore(api_key: str):
    """
    Load existing FAISS vector store from disk.

    Args:
        api_key: OpenAI API key for embeddings

    Returns:
        FAISS vectorstore object
    """
    os.environ["OPENAI_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"[Embeddings] Vectorstore loaded from '{VECTORSTORE_PATH}'")
    return vectorstore
