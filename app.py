import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

print("App is running...")

# ---------------------------------------------------
#  CLEANING FUNCTION
# ---------------------------------------------------

def enhanced_clean_content(text):
    """More aggressive cleaning to remove references & non-research lines."""
    
    reference_patterns = [
        r'REFERENCES.*',
        r'BIBLIOGRAPHY.*',
        r'Reference[s]?.*',
        r'Works Cited.*',
        r'Literature Cited.*',
    ]
    
    for pattern in reference_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        l = line.strip()

        if (re.match(r'^\[\w+\d{2,4}\]', l) or
            re.match(r'^\w+ et al\.', l) or
            re.match(r'^\w+, \w+\.', l) or
            len(l.split()) > 8 and not l.endswith('.') and len(l) < 200 or
            len(l) < 40):
            continue

        if len(l) > 50:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

# ---------------------------------------------------
# PDF INGESTION + CHUNKING
# ---------------------------------------------------

def ingest_and_clean_pdfs(pdf_dir):
    print(f" Searching PDFs in: {pdf_dir}")

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    print(f" Found {len(pdf_files)} PDF files.\n")
    if not pdf_files:
        return [], []

    all_docs = []
    failed_files = []

    for path in pdf_files:
        filename = os.path.basename(path)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()

            clean_docs = []
            for doc in docs:
                clean = enhanced_clean_content(doc.page_content)
                if len(clean.strip()) > 200:
                    doc.page_content = clean
                    clean_docs.append(doc)

            all_docs.extend(clean_docs)
            print(f" {filename}: Loaded {len(clean_docs)} clean pages")

        except Exception as e:
            print(f" {filename}: ERROR — {type(e).__name__}: {e}")
            failed_files.append(filename)

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    chunks = splitter.split_documents(all_docs)
    print(f"\n Total chunks: {len(chunks)}")

    return all_docs, chunks

# ---------------------------------------------------
# VECTOR DATABASE
# ---------------------------------------------------

def build_vector_db(chunks):
    persist_dir = "data/chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f" Vector DB saved to: {persist_dir}")
    return vectordb

# ---------------------------------------------------
# RAG QUERY ENGINE
# ---------------------------------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=300,
    temperature=0.3,
    do_sample=True,
)

def smart_rag_query(question, vectordb, num_docs=8):

    docs = vectordb.similarity_search(question, k=num_docs)
    
    if not docs:
        return {"answer": "No relevant documents found.", "sources": []}

    context_parts = []
    for i, doc in enumerate(docs):
        content = doc.page_content
        lines = content.split("\n")
        clean_lines = [l.strip() for l in lines if len(l.strip()) > 30]
        snippet = " ".join(clean_lines[:5])

        if snippet:
            context_parts.append(f"Document {i+1}:\n{snippet}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an academic research assistant.
Using the context below, answer the question clearly.

CONTEXT:
{context}

QUESTION: {question}

Answer only using the above content.
"""

    try:
        response = generator(prompt, max_length=250)[0]['generated_text']
        return {"answer": response, "sources": docs}
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": docs}

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":

    PDF_DIR = os.path.join(os.getcwd(), "data")
    PDF_DIR = os.path.normpath(PDF_DIR)

    all_docs, chunks = ingest_and_clean_pdfs(PDF_DIR)
    vectordb = build_vector_db(chunks)

    print("\n RAG System Ready!")
    print("Type 'quit' to exit.")
    print("-" * 60)

    while True:
        q = input("\nYour question: ").strip()

        if q.lower() == "quit":
            print("Goodbye!")
            break

        if not q:
            continue

        result = smart_rag_query(q, vectordb)
        print("\nANSWER:\n", result["answer"])
