# TSI-LLM Project: Exploring Natural Language Processing and Machine Learning

This project, **TSI-LLM**, is a collection of Jupyter notebooks and Python scripts designed to explore various aspects of Natural Language Processing (NLP), Machine Learning (ML), and Large Language Models (LLMs). It encompasses a range of tasks, from text embedding and data parsing to prompt engineering and working with open-source models. Additionally, it delves into Retrieval-Augmented Generation (RAG) operations and vector database management.

## Project Features

*   **Text Embedding:** Learn how to generate meaningful vector representations of text data.
*   **Data Parsing:** Explore techniques for extracting and processing information from various data formats.
*   **Prompt Engineering:** Discover best practices for crafting effective prompts for LLMs.
*   **Open-Source Models:** Get hands-on experience with utilizing pre-trained, open-source models for different NLP tasks.
*   **Vector Databases:** Understand how to create, populate, and query vector databases for efficient information retrieval.
*   **Retrieval-Augmented Generation (RAG):** Implement and experiment with RAG techniques for enhancing LLM output.

## Project Structure

The project is organized into the following key directories:

*   **Jupyter notebooks:** All notebooks are stored at the repository root and form the core of the project's educational and experimental content.
    *   Examples:
        *   `langchain-ollama-un02.ipynb`: demonstrates using LangChain with local models.
        *   `embedding-mxbai-un03.ipynb`: explores embedding texts.
        *   `prompt-un06.ipynb`: contains prompt engineering examples.
        *   `test_rag-un08.ipynb`: shows a simple RAG experiment.

*   **`rag/`:** This directory houses Python scripts dedicated to RAG operations. These scripts provide the functionality for implementing and testing various RAG strategies.
    *   Examples:
        * `retrieval.py`: contains the main logic to retrieve information from the database.
        * `generation.py`: contains the logic for generating responses using the LLM.

*   **`data/`:** This directory stores the various data files used throughout the project. These include:
    *   PDF documents
    *   Image files
    *   JSON files

## Getting Started

To get started with this project, clone the repository and explore the notebooks and scripts. You'll need to have a Python environment set up with the necessary dependencies (listed in `requirements.txt`, if provided). Open the notebooks in Jupyter and follow along with the code examples. 

Enjoy exploring the fascinating world of NLP and ML with TSI-LLM!

---

# Projeto TSI-LLM: Explorando Processamento de Linguagem Natural e Machine Learning

Este projeto, **TSI-LLM**, é uma coleção de notebooks Jupyter e scripts Python projetados para explorar vários aspectos do Processamento de Linguagem Natural (NLP), Machine Learning (ML) e Grandes Modelos de Linguagem (LLMs). Ele abrange uma variedade de tarefas, desde incorporação de texto e análise de dados até engenharia de prompts e trabalho com modelos de código aberto. Além disso, ele investiga operações de Geração Aumentada por Recuperação (RAG) e gerenciamento de banco de dados vetorial.

## Recursos do Projeto

*   **Incorporação de Texto:** Aprenda como gerar representações vetoriais significativas de dados de texto.
*   **Análise de Dados:** Explore técnicas para extrair e processar informações de vários formatos de dados.
*   **Engenharia de Prompts:** Descubra as melhores práticas para criar prompts eficazes para LLMs.
*   **Modelos de Código Aberto:** Obtenha experiência prática com a utilização de modelos pré-treinados e de código aberto para diferentes tarefas de NLP.
*   **Bancos de Dados Vetoriais:** Entenda como criar, popular e consultar bancos de dados vetoriais para recuperação eficiente de informações.
*   **Geração Aumentada por Recuperação (RAG):** Implemente e experimente técnicas RAG para aprimorar a saída de LLMs.

## Estrutura do Projeto

O projeto está organizado nos seguintes diretórios-chave:

*   **Notebooks Jupyter:** Os notebooks ficam na raiz do repositório e são o núcleo do conteúdo educacional e experimental do projeto.
    *   Exemplos:
        *   `langchain-ollama-un02.ipynb`: demonstra o uso do LangChain com modelos locais.
        *   `embedding-mxbai-un03.ipynb`: explora incorporação de textos.
        *   `prompt-un06.ipynb`: contém exemplos de engenharia de prompts.
        *   `test_rag-un08.ipynb`: mostra um experimento simples de RAG.

*   **`rag/`:** Este diretório abriga scripts Python dedicados a operações RAG. Esses scripts fornecem a funcionalidade para implementar e testar várias estratégias RAG.
    *   Exemplos:
        *   `retrieval.py`: contém a lógica principal para recuperar informações do banco de dados.
        *   `generation.py`: contém a lógica para gerar respostas usando o LLM.

*   **`data/`:** Este diretório armazena os vários arquivos de dados usados em todo o projeto. Isso inclui:
    *   Documentos PDF
    *   Arquivos de imagem
    *   Arquivos JSON

## Primeiros Passos

Para começar com este projeto, clone o repositório e explore os notebooks e scripts. Você precisará ter um ambiente Python configurado com as dependências necessárias (listadas em `requirements.txt`, se fornecido). Abra os notebooks no Jupyter e acompanhe os exemplos de código.

Aproveite a exploração do fascinante mundo de NLP e ML com TSI-LLM!

## Running Tests

The repository now includes simple unit tests for the RAG modules. After
installing the required dependencies, run them with:

```bash
pytest
```

The tests exercise the ingestion, retrieval and answer generation functions.
