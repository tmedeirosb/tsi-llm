import os
import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_community")
pytest.importorskip("langchain_ollama")

from rag.ingestion import ingest_data, clear_db
from rag.retrieval import load_db, get_relevant_docs
from rag.generation import generate_answer

PDF1 = "data/PPC_TSI_EaD.pdf"
INDEX = "faiss_index"


def test_ingest_create_or_load():
    clear_db(INDEX)
    db = ingest_data(pdf_path=PDF1, index_path=INDEX, update_existing_db=True)
    assert os.path.exists(INDEX)
    assert db.index.ntotal > 0
    db2 = ingest_data(pdf_path=PDF1, index_path=INDEX, update_existing_db=False)
    assert db2.index.ntotal == db.index.ntotal


def test_get_relevant_docs_returns_docs():
    db = load_db(index_path=INDEX)
    docs = get_relevant_docs(db, query="Quais são as disciplinas do curso?", k=5)
    assert len(docs) > 0


def test_generate_answer_not_empty():
    db = load_db(index_path=INDEX)
    docs = get_relevant_docs(db, query="Quais são as disciplinas do curso?", k=5)
    answer = generate_answer(docs, "Quais são as disciplinas do curso?")
    assert isinstance(answer, str)
    assert answer.strip() != ""
