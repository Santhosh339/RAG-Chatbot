from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load a PDF file and split it into chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Number of characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of Document chunks
    """
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"[Loader] Loaded {len(documents)} pages from '{pdf_path}'")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    print(f"[Loader] Split into {len(chunks)} chunks")
    return chunks
