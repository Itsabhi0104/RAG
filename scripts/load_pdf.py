from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def load_pdf(path: str) -> List[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    loader = PyPDFLoader(str(p))
    return loader.load()

if __name__ == "__main__":
    docs = load_pdf(r"F:\RAG\data\Dsa.pdf")

