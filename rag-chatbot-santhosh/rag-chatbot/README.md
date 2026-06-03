# 📄 RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask natural language questions about it. Built with LangChain, OpenAI, FAISS, and Streamlit.

---

## 🚀 Demo

Upload a PDF → Ask questions → Get accurate, context-aware answers with source references.

---

## 🧠 How It Works

```
PDF Upload
    │
    ▼
PyPDFLoader  ──►  Text Splitter (chunks)
                        │
                        ▼
              OpenAI Embeddings (ada-002)
                        │
                        ▼
              FAISS Vector Store (saved locally)
                        │
    User Question ──►  Semantic Search (Top-K retrieval)
                        │
                        ▼
              Relevant Chunks + Question ──► GPT-3.5 Turbo
                        │
                        ▼
                   Final Answer
```

**Key concepts demonstrated:**
- **RAG (Retrieval-Augmented Generation)** — grounding LLM responses in real documents
- **Vector Embeddings** — converting text to numerical vectors for semantic search
- **FAISS** — Facebook AI Similarity Search for fast nearest-neighbor lookup
- **LangChain** — orchestrating the entire retrieval + generation pipeline

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | OpenAI GPT-3.5 Turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | FAISS (faiss-cpu) |
| RAG Framework | LangChain |
| PDF Parsing | PyPDFLoader |

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                  # Streamlit UI — main entry point
├── src/
│   ├── __init__.py
│   ├── loader.py           # PDF loading and text splitting
│   ├── embeddings.py       # Vectorstore creation and loading
│   └── chain.py            # RAG chain (retriever + LLM + prompt)
├── data/                   # Uploaded PDFs stored here (gitignored)
├── vectorstore/            # FAISS index saved here (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 💡 Usage

1. Enter your **OpenAI API Key** in the sidebar
2. Upload any **PDF file**
3. Click **Process Document** — wait for chunking & embedding
4. Type your question in the chat box
5. Get an AI-generated answer with **source chunk references**

---

## 📌 Key Features

- ✅ Upload any PDF (research papers, manuals, books, reports)
- ✅ Semantic search with FAISS — finds the most relevant passages
- ✅ Multi-turn conversation with chat history
- ✅ Source transparency — shows which chunks were used
- ✅ Clean Streamlit UI

---

## 🔮 Future Improvements

- [ ] Support multiple PDFs simultaneously
- [ ] Use open-source LLMs (Llama 3, Mistral) via Hugging Face
- [ ] Add ChromaDB as alternative vector store
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Add document summarization feature

---

## 👤 Author

**A. Santhosh**
B.E. Artificial Intelligence & Machine Learning
Sri Krishna Institute of Technology, Bangalore

- GitHub: [github.com/santhosh](https://github.com/santhosh)
- LinkedIn: [linkedin.com/in/santhosh](https://linkedin.com/in/santhosh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
