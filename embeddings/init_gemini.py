# embeddings/init_gemini.py
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_gemini_embeddings(
    model: str = "models/embedding-001",
    google_api_key: Optional[str] = None,
) -> GoogleGenerativeAIEmbeddings:
    """
    Return a configured GoogleGenerativeAIEmbeddings client using embedding-001 by default.
    """
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_api_key or GEMINI_API_KEY)
