# Agentic RAG — Research Assistant

A Research Assistant system built using **LangChain**, **ChromaDB**, and **FLAN-T5**.  
It ingests PDFs, cleans academic text, builds a vector database, and answers questions using an enhanced RAG pipeline.

---

##  Features

### 🔹 PDF Ingestion + Cleaning
- Loads all PDFs from the `data/` folder  
- Removes:
  - References / Bibliography  
  - Author lists  
  - Citation lines  
  - Very short/noisy lines  
- Keeps only meaningful research text

### 🔹 Smart Text Chunking
- Uses `RecursiveCharacterTextSplitter`
- `chunk_size = 800`
- `chunk_overlap = 150`

### 🔹 Vector Database (ChromaDB)
- Embeddings from: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB saved at:


### 🔹 Enhanced RAG Query System
- Retrieves **8** relevant text chunks  
- Cleans and filters content  
- FLAN-T5 generates final answer  
- Returns sources for each answer

 
---

##  Project Structure

---

## 🔧 Installation

Install dependencies:

```bash
pip install langchain chromadb openai pypdf transformers sentence-transformers torch

