"""
ingest.py — Load documents from /data/docs, chunk, embed, and save a FAISS index.

Supported formats: PDF, TXT
Run once after adding or changing documents:
    python ingest.py
"""

import os
import re
import sys
import pathlib

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCS_DIR = pathlib.Path("data/docs")
INDEX_DIR = pathlib.Path("data/index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_documents():
    """Walk /data/docs and return LangChain Document objects for every PDF and TXT."""
    if not DOCS_DIR.exists():
        print(f"ERROR: {DOCS_DIR} does not exist. Create it and add legal documents.")
        sys.exit(1)

    files = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".txt"}]
    if not files:
        print(f"ERROR: No PDF or TXT files found in {DOCS_DIR}.")
        sys.exit(1)

    all_docs = []
    for path in files:
        print(f"  Loading {path.name} ...", end="", flush=True)
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")

            pages = loader.load()
            for doc in pages:
                # Normalise the act name from the filename (e.g. "rti_act_2005.pdf" -> "Rti Act 2005")
                doc.metadata.setdefault("source", path.name)
                doc.metadata.setdefault("act_name", path.stem.replace("_", " ").title())
            all_docs.extend(pages)
            print(f" {len(pages)} page(s)")
        except Exception as exc:
            print(f" SKIPPED — {exc}")

    return all_docs


def extract_section_hint(text: str) -> str:
    """
    Scan the first 200 characters of a chunk for a section reference.
    Returns an empty string when nothing is found.
    """
    match = re.search(r"(section|sec\.?|§)\s*(\d+[\w.\-]*)", text[:200], re.IGNORECASE)
    return match.group(0).strip() if match else ""


def chunk_and_enrich(docs):
    """Split documents into overlapping chunks and tag each with a section hint."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        hint = extract_section_hint(chunk.page_content)
        if hint:
            chunk.metadata["section"] = hint
    return chunks


def build_and_save_index(chunks):
    """Embed chunks with a local sentence-transformer model and persist FAISS index."""
    print(f"\nEmbedding {len(chunks)} chunks using {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Index saved to {INDEX_DIR}/")


def main():
    print("=== AI Justice Department Assistant — Document Ingestion ===\n")
    print("Loading documents ...")
    docs = load_documents()
    print(f"\nLoaded {len(docs)} page(s) in total.")

    chunks = chunk_and_enrich(docs)
    print(f"Split into {len(chunks)} chunks  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    build_and_save_index(chunks)
    print("\nIngestion complete. You can now start the server.")


if __name__ == "__main__":
    main()

