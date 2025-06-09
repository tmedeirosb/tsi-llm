# __init__.py
"""
Pacote RAG
----------
Arquivo de inicialização do pacote 'rag'.
"""

from .ingestion import ingest_data, clear_db
from .retrieval import load_db, get_relevant_docs
from .generation import generate_answer
from .youtube_agents import TranscriptAgent, SummaryQAAgent
