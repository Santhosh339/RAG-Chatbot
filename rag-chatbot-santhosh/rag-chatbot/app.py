import os
import streamlit as st
from src.loader import load_and_split_pdf
from src.embeddings import build_vectorstore, load_vectorstore
from src.chain import build_rag_chain

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 RAG Document Q&A Chatbot")
st.markdown("Upload a PDF and ask questions about it using AI.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    process_btn = st.button("Process Document", type="primary")
    st.markdown("---")
    st.markdown("**Built by A. Santhosh**")
    st.markdown("AI/ML Fresher | Gen AI Developer")

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# ── Process PDF ───────────────────────────────────────────────────────────────
if process_btn and uploaded_file:
    if not api_key:
        st.sidebar.error("Please enter your OpenAI API key.")
    else:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            pdf_path = f"data/{uploaded_file.name}"
            os.makedirs("data", exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load, split, embed
            chunks = load_and_split_pdf(pdf_path)
            build_vectorstore(chunks, api_key)
            st.session_state.vectorstore_ready = True
            st.session_state.chat_history = []
            st.sidebar.success(f"✅ Processed {len(chunks)} chunks!")

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.vectorstore_ready:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = build_rag_chain(api_key)
                response = chain.invoke({
                    "input": user_input,
                    "chat_history": [
                        (m["content"] if m["role"] == "user" else "")
                        for m in st.session_state.chat_history[:-1]
                    ]
                })
                answer = response["answer"]
                st.write(answer)

                # Show source documents
                with st.expander("📚 Source Chunks Used"):
                    for i, doc in enumerate(response.get("context", []), 1):
                        st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}...")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF from the sidebar and click **Process Document** to get started.")
