from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split a list of langchain Documents into smaller chunk Documents.

    - chunk_size and chunk_overlap are in characters.
    - Returns new Document objects with preserved metadata + 'chunk' index.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: List[Document] = []
    for d in docs:
        text = d.page_content or ""
        parts = splitter.split_text(text)
        base_meta = dict(d.metadata or {})
        for i, part in enumerate(parts, start=1):
            meta = base_meta.copy()
            meta["chunk"] = i
            out.append(Document(page_content=part, metadata=meta))
    return out
