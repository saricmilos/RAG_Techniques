from typing import List
from langchain_core.documents import Document

def clean_document_text(documents: List[Document]) -> List[Document]:
    """Replaces tab characters with spaces using modern string expansion."""
    for doc in documents:
        doc.page_content = doc.page_content.expandtabs(1)
    return documents