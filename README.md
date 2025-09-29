# Startup Business Proposal RAG Application

This project implements a Retrieval-Augmented Generation (RAG) application designed to answer questions about startup business proposals using a collection of PDF documents.

## Features

* **Document Processing:** Loads PDF documents and chunks them by paragraph.
* **Embedding Generation:** Uses the BGE-base-en-v1.5 model to create vector embeddings of document chunks.
* **Vector Database:** Stores and retrieves document embeddings using Qdrant (with Cosine similarity).
* **Language Model Integration:** Leverages Groq's LLM (e.g., Llama3-8B) for generating answers.
* **Prompting Techniques:** Employs context-injection and optional few-shot examples for improved answer generation.
* **Evaluation:** Calculates F1-score to quantify the accuracy of generated responses against ground truth.

## Assigned Methods

* **Documents:** 10 PDF documents related to startup business proposals.
* **Chunk Method:** Paragraph chunking using `RecursiveCharacterTextSplitter`.
* **Embeddings:** BGE-base-en-v1.5 (Hugging Face) via `sentence-transformers`.
* **Vector DB & Indexing:** Qdrant (Cosine similarity).
* **Prompting Techniques:** Few-shot, context-injection.
* **LLM:** Groq API.
* **Evaluation:** F1-Score.

## Setup

### 1. Clone the Repository (or create your project directory)

```bash
git clone <your_repo_url>
cd startup_rag_app
