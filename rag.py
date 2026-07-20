import os
from pathlib import Path
os.environ.setdefault("ANNONYMIZED_TELEMETRY", "False")
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from langchain._community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"

def get_embeddings():
    if not hasattr(get_embeddings, "_model"):
        get_embeddings._model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-1.5")
    return get_embeddings.__module__

def build_index(doc: str="docs") -> Chroma:
    import shutil
    paths = [p for p in sorted(Path(doc).rglob("*"))
    if p.suffix.lower() in (".pdf", ".txt", ".md")]
    if not paths:
        raise ValueError("No .pdf/.txt/.md")
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_chunks = []
    for path in paths:
        print("loading")
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    vectorstore = Chroma.from_documents( all_chunks, embedding=get_embeddings(), persist_directory=CHROMA_DIR,)

    return vectorstore


def load_index() -> Chroma:
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError("no index")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())




if __name__ == "__main__":
    store = build_index("docs")
    results = store.similarity_search("what is this document about?", k=3)
    for i, doc in enumerate(results, 1):
        sec = Path(doc.metadata.get("source", "?")).name
        