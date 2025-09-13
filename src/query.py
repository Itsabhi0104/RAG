# query.py
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
import pyreadline3 as readline
import time

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "rag")

if not GEMINI_KEY or not PINECONE_API_KEY:
    raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) and PINECONE_API_KEY in .env")

# Initialize Pinecone with the new API
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

emb = GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
                                  google_api_key=GEMINI_KEY)

llm_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0, google_api_key=GEMINI_KEY)

history = []

def rewrite_followup(question):
    """
    Use Gemini to rewrite follow-ups into stand-alone question.
    """
    system = SystemMessage(content=(
        "You are a query rewriting expert. "
        "Based on the provided chat history, rephrase the 'Follow Up user Question' into a complete, standalone question. "
        "Only output the rewritten question and nothing else."
    ))
    
    context = "\n".join([f"User: {m}" for m in history[-6:]])  # last few messages
    human_text = f"Chat history:\n{context}\n\nFollow Up user Question: {question}"
    human = HumanMessage(content=human_text)
   
    resp = llm.invoke([system, human])
    out = getattr(resp, "content", None) or getattr(resp, "message", None) or str(resp)
    return out.strip()

def retrieve_context(query, top_k=10):
    qvec = emb.embed_query(query)
    resp = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    matches = resp.matches if hasattr(resp, "matches") else resp["matches"]
    snippets = []
    for m in matches:
        md = m.metadata or {}
        text = md.get("text") or md.get("content") or ""
        source = md.get("source", "")
        snippets.append(f"Source: {source}\n{text}\n")
    context = "\n\n---\n\n".join(snippets)
    return context, matches

def answer_question(question):
    if question.strip().lower().startswith(("what about", "and", "also", "follow up", "fyi")) and history:
        question = rewrite_followup(question)

    context, matches = retrieve_context(question, top_k=10)
    system_instruction = (
        "You have to behave like a Data Structure and Algorithm Expert.\n"
        "You will be given a context of relevant information and a user question.\n"
        "Your task is to answer the user's question based ONLY on the provided context.\n"
        "If the answer is not in the context, you must say \"I could not find the answer in the provided document.\"\n"
        "Keep your answers clear, concise, and educational.\n\n"
        f"Context:\n{context}"
    )

    system = SystemMessage(content=system_instruction)
    human = HumanMessage(content=question)
    resp = llm.invoke([system, human])
    out = getattr(resp, "content", None) or getattr(resp, "message", None) or str(resp)

    # save history
    history.append(question)
    history.append(out)
    return out

def main_loop():
    print("RAG CLI — ask questions about the DSA PDF. Ctrl-C to exit.")
    while True:
        try:
            q = input("\nAsk me anything--> ").strip()
            if not q:
                continue
            answer = answer_question(q)
            print("\n---\n")
            print(answer)
            print("\n---\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main_loop()