"""
app.py — FastAPI server.

Endpoints:
  GET  /               → serves static/index.html
  POST /ask            → RAG question answering (non-streaming)
  POST /ask/stream     → RAG question answering (SSE streaming)
  POST /feedback       → append helpful/not-helpful vote to feedback.csv
  GET  /evaluate       → show evaluation results table (if results.csv exists)
  GET  /health         → health check
"""

import csv
import json
import pathlib
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from rag import answer_question, get_vectorstore, stream_answer

STATIC_DIR = pathlib.Path("static")
FEEDBACK_CSV = pathlib.Path("data/feedback.csv")
RESULTS_CSV = pathlib.Path("data/eval/results.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_vectorstore()
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")
    yield


app = FastAPI(
    title="AI Justice Department Assistant",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


# ── Request / response models ─────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []
    language: str = "en"

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Question is too long. Keep it under 2000 characters.")
        return v

    @field_validator("language")
    @classmethod
    def language_valid(cls, v: str) -> str:
        if v not in ("en", "hi"):
            return "en"
        return v


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(body: AskRequest):
    history = [{"question": t.question, "answer": t.answer} for t in body.history]
    try:
        result = answer_question(body.question, history=history, language=body.language)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="The document index is not ready. Run python ingest.py first.",
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="The assistant encountered an error. Please try again.",
        )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "citations": result["citations"],
        "standalone_question": result["standalone_question"],
    }


@app.post("/ask/stream")
def ask_stream(body: AskRequest):
    """
    Server-sent events stream. Each event has a JSON payload:
      data: {"type": "status"|"sources"|"token"|"done"|"error", "data": ...}
    """
    history = [{"question": t.question, "answer": t.answer} for t in body.history]

    def event_generator():
        try:
            for event_type, data in stream_answer(
                body.question, history=history, language=body.language
            ):
                payload = json.dumps({"type": event_type, "data": data})
                yield f"data: {payload}\n\n"
        except Exception:
            traceback.print_exc()
            payload = json.dumps({
                "type": "error",
                "data": "The assistant encountered an error while streaming. Please try again.",
            })
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "helpful", "question", "answer"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "helpful": body.helpful,
            "question": body.question[:500],
            "answer": body.answer[:1000],
        })
    return {"status": "recorded"}


@app.get("/evaluate", response_class=HTMLResponse)
def evaluate_page():
    if not RESULTS_CSV.exists():
        return HTMLResponse(
            "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Evaluation Results</title>"
            "<style>body{font-family:sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1c1c1c;background:#f8f7f5;}"
            "a{color:#1a3a5c;}</style></head><body>"
            "<h1>Evaluation Results</h1>"
            "<p>No evaluation results found. Place evaluation QA data in <code>data/eval/</code> and run <code>python evaluate.py</code> first.</p>"
            "<p><a href='/'>Back to assistant</a></p></body></html>",
            status_code=404,
        )

    rows = []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return HTMLResponse("<html><body><p>results.csv is empty.</p><p><a href='/'>Back</a></p></body></html>")

    headers = list(rows[0].keys())
    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = ""
    for row in rows:
        cells = "".join(f"<td>{row.get(h, '')}</td>" for h in headers)
        body_rows += f"<tr>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Evaluation Results — AI Justice Department Assistant</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1c1c1c; background: #f8f7f5; line-height: 1.5; }}
  h1 {{ font-family: Georgia, serif; font-size: 1.4rem; font-weight: normal; border-bottom: 1px solid #c8c4bc; padding-bottom: .5rem; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: 1rem; background: #fff; }}
  th, td {{ border: 1px solid #c8c4bc; padding: .5rem .6rem; text-align: left; vertical-align: top; }}
  th {{ background: #eceae4; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fdfcfb; }}
  a {{ color: #1a3a5c; text-decoration: underline; }}
</style>
</head>
<body>
<h1>Evaluation Results</h1>
<p><a href="/">Back to assistant</a></p>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>{body_rows}</tbody>
</table>
</body>
</html>"""
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(html_path))
