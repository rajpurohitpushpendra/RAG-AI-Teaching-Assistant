from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import UPLOAD_DIR
from services.rag_service import answer_question, process_pdf

FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(title="PDF RAG Teaching Assistant")


class HistoryMessage(BaseModel):
    role: str
    content: str = Field(max_length=1500)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filename: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)


@app.get("/")
def home():
    return FileResponse(FRONTEND_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    files = sorted(UPLOAD_DIR.glob("*.pdf"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {
        "documents": [
            {"filename": path.name, "size_bytes": path.stat().st_size}
            for path in files
        ]
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file")

    content = await file.read()
    await file.close()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF must be under 20 MB")

    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    filename = f"{uuid4().hex}.pdf"
    destination = UPLOAD_DIR / filename
    destination.write_bytes(content)

    try:
        chunks = process_pdf(filename)
        return {
            "message": "PDF processed",
            "filename": filename,
            "original_name": Path(file.filename).name,
            "chunks_indexed": chunks,
        }
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not process PDF: {exc}",
        ) from exc


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if Path(request.filename).name != request.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not (UPLOAD_DIR / request.filename).is_file():
        raise HTTPException(
            status_code=404,
            detail="Selected PDF not found. Please upload it again.",
        )

    history = [item.model_dump() for item in request.history]

    try:
        return answer_question(question, request.filename, history)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not answer question. Check that the PDF is indexed "
                "and Ollama is running."
            ),
        ) from exc
