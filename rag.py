"""
rag.py — RAG pipeline: retrieval, LLM answering, streaming, history handling.
Imported by app.py and evaluate.py.
"""

import os
import pathlib
import traceback

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_DIR = pathlib.Path("data/index")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))

NO_ANSWER = "I couldn't find this in the available legal documents."

# System prompt for answering legal questions.
SYSTEM_PROMPT = """You are a legal reference assistant for Indian law.
Answer the user's question using ONLY the numbered context passages provided below.
When you make a claim, cite it with the passage number in square brackets, e.g. [1] or [2].
Use plain language. Keep sentences short. Do not speculate beyond the context.
If the context does not contain enough information to answer, respond with exactly:
"I couldn't find this in the available legal documents."
Do not add greetings, disclaimers, or commentary outside the answer.
Always keep act names and section numbers in their original English form in citations."""

# Prompt to rewrite follow-up questions into standalone queries.
REWRITE_PROMPT = """Given the conversation history below and a follow-up question,
rewrite the follow-up as a fully self-contained question that can be answered without
seeing the history. Output only the rewritten question, nothing else.

History:
{history}

Follow-up: {question}
Standalone question:"""


# ── Singletons ───────────────────────────────────────────────────────────────

_vectorstore = None
_llm = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _load_vectorstore()
    return _vectorstore


def get_llm():
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_vectorstore():
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        raise FileNotFoundError(
            f"No index found at {INDEX_DIR}. Run `python ingest.py` first."
        )
    embeddings = _load_embeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use openai, gemini, or ollama.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = (
            f"[{i}] Act: {m.get('act_name', 'Unknown')} | "
            f"Section: {m.get('section', 'N/A')} | "
            f"File: {m.get('source', 'N/A')} | "
            f"Page: {m.get('page', 'N/A')}"
        )
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _build_sources(docs, scores=None) -> list[dict]:
    """Build source objects, deduplicating by (act, section, file, page)."""
    sources = []
    seen = set()
    for i, doc in enumerate(docs):
        m = doc.metadata
        key = (m.get("act_name"), m.get("section"), m.get("source"), m.get("page"))
        if key in seen:
            continue
        seen.add(key)
        source = {
            "index": len(sources) + 1,
            "act_name": m.get("act_name", "Unknown Act"),
            "section": m.get("section", "N/A"),
            "file": m.get("source", "N/A"),
            "page": m.get("page", "N/A"),
            "passage": doc.page_content.strip(),
        }
        if scores is not None and i < len(scores):
            source["score"] = round(float(scores[i]), 4)
        sources.append(source)
    return sources


def _rewrite_question(question: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up question into a self-contained one using conversation history.
    Only called when there is prior history. Falls back to original on error.
    """
    if not history:
        return question

    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in history[-4:]
    )
    prompt = REWRITE_PROMPT.format(history=history_text, question=question)
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        return rewritten if rewritten else question
    except Exception as exc:
        print(f"Warning: question rewrite failed ({exc}); using original.")
        return question


def _build_prompt(question: str, context: str, language: str) -> str:
    lang_instruction = ""
    if language and language.lower() == "hi":
        lang_instruction = "\nAnswer in Hindi. Keep all act names and section numbers in English."
    return f"{SYSTEM_PROMPT}{lang_instruction}\n\nContext:\n{context}\n\nQuestion: {question}"


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K):
    """Return (docs, scores) from the vector store."""
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(question, k=top_k)
    docs = [r[0] for r in results]
    scores = [r[1] for r in results]
    return docs, scores


def answer_question(
    question: str,
    history: list[dict] | None = None,
    language: str = "en",
    top_k: int = TOP_K,
) -> dict:
    """
    Full non-streaming RAG round-trip.
    """
    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "citations": [],
            "retrieved_docs": [],
            "standalone_question": "",
        }

    history = history or []
    standalone = _rewrite_question(question, history) if history else question
    docs, scores = retrieve(standalone, top_k)

    if not docs:
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "citations": [],
            "retrieved_docs": [],
            "standalone_question": standalone,
        }

    context = _format_context(docs)
    prompt = _build_prompt(standalone, context, language)

    llm = get_llm()
    response = llm.invoke(prompt)
    answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    if not answer_text:
        answer_text = NO_ANSWER

    sources = _build_sources(docs, scores)
    return {
        "answer": answer_text,
        "sources": sources,
        "citations": sources,
        "retrieved_docs": docs,
        "standalone_question": standalone,
    }


def stream_answer(
    question: str,
    history: list[dict] | None = None,
    language: str = "en",
    top_k: int = TOP_K,
):
    """
    Generator that yields (event_type, data) tuples for SSE streaming.
    event_type is one of: "status", "sources", "token", "done", "error"
    """
    question = question.strip()
    if not question:
        yield ("error", "Please enter a question.")
        return

    history = history or []

    try:
        yield ("status", "searching")
        standalone = _rewrite_question(question, history) if history else question

        docs, scores = retrieve(standalone, top_k)

        if not docs:
            yield ("sources", [])
            yield ("token", NO_ANSWER)
            yield ("done", "")
            return

        sources = _build_sources(docs, scores)
        yield ("sources", sources)
        yield ("status", "writing")

        context = _format_context(docs)
        prompt = _build_prompt(standalone, context, language)

        llm = get_llm()
        full_text = ""
        for chunk in llm.stream(prompt):
            if hasattr(chunk, "content"):
                c = chunk.content
                if isinstance(c, str):
                    token = c
                elif isinstance(c, list):
                    token = "".join(part if isinstance(part, str) else part.get("text", "") for part in c)
                else:
                    token = str(c)
            else:
                token = str(chunk)

            if token:
                full_text += token
                yield ("token", token)

        if not full_text.strip():
            yield ("token", NO_ANSWER)

        yield ("done", "")

    except FileNotFoundError:
        yield ("error", "The document index is not ready. Run python ingest.py first.")
    except Exception:
        traceback.print_exc()
        yield ("error", "The assistant encountered an error while processing your question. Please try again.")
