# generation.py
"""
Módulo de Geração (Generation)
------------------------------
Este módulo é responsável por:
1. Receber documentos relevantes (chunks de texto).
2. Usar um modelo de linguagem (LLM) para gerar respostas com base nesses documentos.

Dependências externas utilizadas:
- langchain_ollama
- langchain
"""

from typing import List
from langchain.docstore.document import Document
from langchain_ollama.llms import OllamaLLM
from langchain import PromptTemplate

def generate_answer(docs: List[Document], question: str) -> str:
    """
    Função principal do módulo de geração que:
      - Recebe uma lista de documentos relevantes.
      - Cria um prompt com base no contexto e na questão do usuário.
      - Chama o modelo OllamaLLM para obter a resposta gerada.

    Parâmetros:
    -----------
    docs : List[Document]
        Lista de documentos relevantes que servirão de contexto.
    question : str
        Pergunta do usuário a ser respondida.

    Retorno:
    --------
    resposta : str
        Texto gerado pelo modelo com base no contexto.
    """

    model = OllamaLLM(model="llama3.2:latest")

    template = """
Responda a questão baseada apenas no texto a seguir:

Contexto: "{contexto}"
Questão: "{questao}"
"""
    prompt = PromptTemplate(input_variables=["contexto", "questao"], template=template)
    contexto = "\n".join([doc.page_content for doc in docs])
    prompt_text = prompt.format(contexto=contexto, questao=question)

    print(f"Prompt gerado: \n{prompt_text}")

    resposta = model.invoke(prompt_text)
    return resposta