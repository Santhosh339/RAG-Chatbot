import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.embeddings import load_vectorstore

# ── Prompt Template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions based strictly on the provided context.

Instructions:
- Answer only using the context below. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that information in the document."
- Be concise, accurate, and cite which part of the document your answer comes from.
- If asked to summarize, provide a clear, structured summary.

Context:
{context}
"""


def build_rag_chain(api_key: str):
    """
    Build the RAG chain: Retriever + LLM + Prompt.

    Args:
        api_key: OpenAI API key

    Returns:
        Runnable RAG chain
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Load vectorstore and create retriever
    vectorstore = load_vectorstore(api_key)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}   # Retrieve top 4 most relevant chunks
    )

    # LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,           # Keep answers factual, not creative
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Build chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("[Chain] RAG chain built successfully.")
    return rag_chain
