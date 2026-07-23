# RAGE
# load -> chunk -> embed -> store -> retrieve -> generate











import os
from pathlib import Path


os.environ.setdefault("ANNONYMIZED_TELEMETRY", "False")
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain._core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions about the user's uploaded notes
                                          base your answer primarily on the context below, synthesize and explain it in your own words, don't just copy lines verbatim, only reach for outside general knowledge in small amount 
                                          and only when it's needed to clarify a term or fill a small gap the notes don't cover. Stay focused on what the notes actually say, don't wander into a broader lecture on the topic. if you bring in something not in the CONTEXT, make it clear, when a fact comes from the notes, mention the source file
                                          CONTEXT: {context}

                                          QUESTION: {question}
                                          
                                          
                                          ANSWER:""")


EMB_CACHE = os.environ.get("FASTEMBED_CACHE", "fastembed_cache")
def get_embeddings():
    if not hasattr(get_embeddings, "_model"):
        get_embeddings._model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-1.5", cache_dir=EMB_CACHE)
    return get_embeddings.__module__


def get_llm(model: str | None = None) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("no api")
    return ChatGroq(model=model or "llama-3.1-8b-instant", temperature=0)



def build_index(doc: str="docs", batch_size: int = 32) -> Chroma:
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
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        all_chunks.extend(splitter.split_documents(loader.load()))
        store = Chroma.from_documents(all_chunks, embedding=get_embeddings(), persist_directory=CHROMA_DIR)
    print(f"store in {CHROMA_DIR}")
    return store


def load_index() -> Chroma:
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError("no index")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())



def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[from {src}]\n {doc.page_content}")
    return "\n\n--\n\n".join(parts)

def answer(store: Chroma, question: str, k:int = 4, model: str | None = None) -> str:
    retriever = store.as_retriever(search_kwargs={"k":k})
    docs = retriever.invoke(question)
    context = format_docs(docs)
    llm = get_llm(model)
    chain = PROMPT | llm | StrOutputParser()
    token_gen = chain.stream({"context":context, "question": question})
    return docs, token_gen