from pathlib import Path
from typing import List, Union

from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_document_text(documents: List[Document]) -> List[Document]:
    """Replaces tab characters with spaces using modern string expansion."""
    for doc in documents:
        doc.page_content = doc.page_content.expandtabs(1)
    return documents

def encode_pdf(
    path: Union[str, Path], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> FAISS:
    """
    Encodes a PDF into a FAISS vector store using local HuggingFace embeddings.
    """
    # 1. Path Handling
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No PDF found at: {pdf_path.absolute()}")

    # 2. Load PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 3. Clean and Split
    # We clean before splitting to ensure whitespace doesn't mess with chunk boundaries
    cleaned_docs = clean_document_text(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True 
    )
    texts = text_splitter.split_documents(cleaned_docs)

    # 4. Local Hugging Face Embeddings
    # This model is ~80MB and runs purely on your machine
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )

    # 5. Create and return Vector Store
    return FAISS.from_documents(texts, embeddings)