# ingestion.py
"""
Módulo de Ingestão (Ingestion)
-----------------------------
Este módulo é responsável por:
1. Carregar o PDF.
2. Dividir o conteúdo em chunks.
3. Criar (ou carregar) e atualizar a base vetorial (FAISS).
4. Oferecer métodos para limpar a base (se necessário).
5. Exibir o tamanho da base após inserções.
"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

def ingest_data(
    pdf_path: str, 
    index_path: str = "faiss_index", 
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    update_existing_db: bool = True
) -> FAISS:
    """
    Função principal do módulo de ingestão que:
      1. Carrega o PDF e divide em chunks.
      2. Verifica se já existe uma base vetorial FAISS salva:
         - Se existir:
             - Se 'update_existing_db' for True, adiciona novos documentos.
             - Caso contrário, apenas carrega a base.
         - Se não existir, cria uma nova base FAISS a partir dos chunks.
      3. Salva/atualiza a base FAISS localmente.
      4. Exibe o tamanho da base após a inserção.

    Parâmetros:
    -----------
    pdf_path : str
        Caminho para o arquivo PDF a ser ingerido.
    index_path : str, opcional
        Caminho (pasta ou arquivo) onde a base FAISS será salva/carregada.
    chunk_size : int, opcional
        Tamanho de cada chunk (número de caracteres).
    chunk_overlap : int, opcional
        Sobreposição entre os chunks.
    update_existing_db : bool, opcional
        Se True, adiciona novos documentos a um DB já existente;
        se False, apenas carrega o DB existente (sem adicionar novos dados).
        Se o DB não existir, sempre criará um novo.

    Retorno:
    --------
    db : FAISS
        Base vetorial contendo embeddings dos documentos processados.
    """

    # 1. Carregando o PDF e dividindo em chunks
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(pages)

    if texts:
        print("Exemplo do primeiro chunk:\n", texts[0])
    print(f"Quantidade de chunks: {len(texts)}")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 2. Verifica se já existe um índice FAISS salvo
    if os.path.exists(index_path):
        print(f"Carregando base FAISS existente em '{index_path}'...")
        db = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True  # Somente se você confia no arquivo
        )

        if update_existing_db and texts:
            # Tamanho antes da inserção
            previous_count = db.index.ntotal  
            print(f"Tamanho atual da base (antes de inserir): {previous_count}")

            print("Atualizando base FAISS com novos documentos...")
            db.add_documents(texts)

            # Tamanho depois da inserção
            new_count = db.index.ntotal
            print(f"Base atualizada! Tamanho antes: {previous_count}, depois: {new_count}")
        else:
            print("Base FAISS carregada. Nenhuma atualização realizada.")
            
            # Exibir o tamanho atual (opcional, se você quiser ver quantos vetores existem)
            current_count = db.index.ntotal
            print(f"Tamanho atual da base: {current_count}")
    else:
        print("Nenhuma base FAISS encontrada. Criando uma nova base...")
        db = FAISS.from_documents(texts, embeddings)

        # Exibir o tamanho do índice recém-criado
        new_count = db.index.ntotal
        print(f"Nova base criada com {new_count} vetores.")

    # 3. Salva/atualiza a base FAISS localmente
    db.save_local(index_path)
    print(f"Base FAISS salva/atualizada em '{index_path}'.")

    return db

def clear_db(index_path: str = "faiss_index"):
    """
    Remove a base FAISS salva localmente, caso exista, limpando assim os embeddings.
    
    Parâmetros:
    -----------
    index_path : str, opcional
        Caminho onde a base FAISS está salva.
    """
    import shutil
    if os.path.exists(index_path):
        if os.path.isdir(index_path):
            shutil.rmtree(index_path)
            print(f"Base FAISS removida: diretório '{index_path}' apagado.")
        else:
            os.remove(index_path)
            print(f"Base FAISS removida: arquivo '{index_path}' apagado.")
    else:
        print(f"Nenhuma base encontrada em '{index_path}' para remover.")