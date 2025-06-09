# retrieval.py
"""
Módulo de Recuperação (Retrieval)
---------------------------------
Este módulo é responsável por:
1. Carregar a base FAISS a partir do disco, sem adicionar dados.
2. Criar um retriever para encontrar os documentos mais relevantes.
3. Retornar documentos relevantes a partir de uma query do usuário.

Dependências externas utilizadas:
- langchain
- langchain_community
"""

from typing import List
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

def load_db(index_path: str = "faiss_index") -> FAISS:
    """
    Carrega o índice FAISS do disco, sem adicionar documentos novos.
    
    Parâmetros:
    -----------
    index_path : str, opcional
        Caminho onde a base FAISS está salva. Por padrão, "faiss_index".
    
    Retorno:
    --------
    db : FAISS
        Instância carregada da base FAISS.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"Base FAISS carregada de '{index_path}'.")
    return db

def get_relevant_docs(db: FAISS, query: str, k: int = 5) -> List[Document]:
    """
    Recebe a instância FAISS em memória e retorna os 'k' documentos mais relevantes
    para a query fornecida.

    Parâmetros:
    -----------
    db : FAISS
        Base FAISS carregada, contendo embeddings dos documentos.
    query : str
        Pergunta (query) do usuário para buscar documentos relevantes.
    k : int, opcional
        Quantidade de documentos relevantes a serem retornados. Padrão é 5.

    Retorno:
    --------
    docs : List[Document]
        Lista dos documentos mais relevantes para a query.
    """
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return docs
