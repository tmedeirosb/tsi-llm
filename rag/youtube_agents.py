"""YouTube Agents
-----------------
This module defines two simple agents:
1. ``TranscriptAgent`` - Receives a YouTube URL, downloads the transcript and
   stores it in a FAISS vector database.
2. ``SummaryQAAgent`` - Summarises the stored transcript, suggests questions and
   answers them using a basic RAG pipeline.

The implementation relies on the existing modules in the ``rag`` package for
retrieval and generation of answers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM

from . import retrieval, generation


@dataclass
class TranscriptAgent:
    """Agent responsible for fetching and storing YouTube transcripts."""

    index_path: str = "faiss_index"
    chunk_size: int = 1000
    chunk_overlap: int = 50

    def _video_id_from_url(self, url: str) -> str:
        """Extract the YouTube video ID from a URL."""
        parsed = urlparse(url)
        if parsed.hostname in {"youtu.be"}:
            return parsed.path.lstrip("/")
        if parsed.hostname and "youtube" in parsed.hostname:
            query = parse_qs(parsed.query)
            return query.get("v", [""])[0]
        return url

    def fetch_transcript(self, url: str) -> str:
        """Download the transcript for ``url`` and return it as plain text."""
        video_id = self._video_id_from_url(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["pt", "en"])
        text = " ".join(item["text"] for item in transcript)
        return text

    def store_transcript(self, text: str) -> FAISS:
        """Split ``text`` and store the chunks in a FAISS index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )
        docs = splitter.split_documents([Document(page_content=text)])
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        if os.path.exists(self.index_path):
            db = FAISS.load_local(
                self.index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            db.add_documents(docs)
        else:
            db = FAISS.from_documents(docs, embeddings)
        db.save_local(self.index_path)
        return db

    def run(self, url: str) -> FAISS:
        """Fetch transcript from ``url`` and store it."""
        text = self.fetch_transcript(url)
        return self.store_transcript(text)


@dataclass
class SummaryQAAgent:
    """Agent that summarises the transcript and answers generated questions."""

    index_path: str = "faiss_index"
    questions: int = 3

    def _load_db(self) -> FAISS:
        return retrieval.load_db(self.index_path)

    def _summarise(self, db: FAISS) -> str:
        # Retrieve all documents for a full summary
        docs = db.similarity_search(" ", k=db.index.ntotal)
        text = "\n".join(doc.page_content for doc in docs)
        model = OllamaLLM(model="llama3.2:latest")
        prompt = (
            "Resuma o texto a seguir em português de forma concisa:\n" + text + "\nResumo:""
        )
        return model.invoke(prompt)

    def _suggest_questions(self, summary: str) -> List[str]:
        model = OllamaLLM(model="llama3.2:latest")
        prompt = (
            f"Com base no resumo a seguir, sugira {self.questions} perguntas que um aluno poderia fazer:\n"
            f"{summary}\nPerguntas:""
        )
        output = model.invoke(prompt)
        return [q.strip("- ") for q in output.split("\n") if q.strip()]

    def _answer(self, db: FAISS, question: str) -> str:
        docs = retrieval.get_relevant_docs(db, question, k=5)
        return generation.generate_answer(docs, question)

    def run(self) -> List[tuple[str, str]]:
        db = self._load_db()
        summary = self._summarise(db)
        questions = self._suggest_questions(summary)
        answers = [(q, self._answer(db, q)) for q in questions]
        return answers
