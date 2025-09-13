import os
import uuid
import time
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "rag")
PDF_PATH = r"F:\RAG\data\Dsa.pdf"  
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
BATCH_SIZE = 32

if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY is missing in env.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in env.")

def load_pdf(path):
    print("Loading PDF:", path)
    loader = PyPDFLoader(path, mode="single")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from PDF.")
    return docs

def chunk_documents(docs):
    print(f"Splitting into chunks: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked = splitter.split_documents(docs)
    print(f"Created {len(chunked)} chunks.")
    return chunked

def init_embeddings():
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    emb = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=GEMINI_KEY)
    return emb

def init_pinecone():
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get the index
    index = pc.Index(PINECONE_INDEX)
    return index

def upsert_chunks(index, embeddings, docs):
    print("Embedding & upserting chunks to Pinecone...")
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metas = metas[i : i + BATCH_SIZE]
        
        vectors = embeddings.embed_documents(batch_texts)  # returns list[list[float]]
        
        to_upsert = []
        for j, vec in enumerate(vectors):
            idx = str(uuid.uuid4())
            meta = batch_metas[j] or {}
            meta.update({"text": batch_texts[j], "source": os.path.basename(PDF_PATH)})
            to_upsert.append((idx, vec, meta))
       
        index.upsert(vectors=to_upsert)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
    
    print("Upsert completed.")

def main():
    docs = load_pdf(PDF_PATH)
    chunked = chunk_documents(docs)
    embeddings = init_embeddings()
    pinecone_index = init_pinecone()
    upsert_chunks(pinecone_index, embeddings, chunked)
    print("Ingest finished.")

if __name__ == "__main__":
    main()