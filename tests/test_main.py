
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_loads():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"This is not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a PDF file"


def test_upload_rejects_invalid_pdf_content():
    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.pdf",
                b"This is not a valid PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF file"


from services import rag_service


def test_process_pdf_splits_and_indexes_document(monkeypatch, tmp_path):
    from langchain_core.documents import Document

    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(rag_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        rag_service.PyPDFLoader,
        "load",
        lambda self: [
            Document(
                page_content="StudyMate AI helps students learn.",
                metadata={"page": 0},
            )
        ],
    )

    class FakeStore:
        def __init__(self):
            self.added_documents = None

        def get(self, **kwargs):
            return {"ids": []}

        def add_documents(self, documents):
            self.added_documents = documents

    fake_store = FakeStore()
    monkeypatch.setattr(
        rag_service, "get_vector_store", lambda: fake_store
    )

    count = rag_service.process_pdf("sample.pdf")

    assert count >= 1
    assert fake_store.added_documents is not None
    assert fake_store.added_documents[0].metadata["source"] == "sample.pdf"


def test_process_pdf_rejects_empty_document(monkeypatch, tmp_path):
    from langchain_core.documents import Document

    fake_pdf = tmp_path / "empty.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(rag_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        rag_service.PyPDFLoader,
        "load",
        lambda self: [
            Document(page_content="   ", metadata={"page": 0})
        ],
    )

    try:
        rag_service.process_pdf("empty.pdf")
        assert False, "Expected ValueError for empty PDF"
    except ValueError as exc:
        assert "No readable text" in str(exc)


def test_answer_question_rejects_empty_question(monkeypatch):
    result = rag_service.answer_question(
        question="   ",
        filename="sample.pdf",
    )

    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []
    assert result["grounded"] is False


def test_answer_question_handles_unindexed_pdf(monkeypatch):
    monkeypatch.setattr(
        rag_service,
        "get_document_chunks",
        lambda filename: [],
    )

    result = rag_service.answer_question(
        question="What is this PDF about?",
        filename="missing.pdf",
    )

    assert "couldn't find indexed text" in result["answer"]
    assert result["sources"] == []
    assert result["grounded"] is False

def test_ask_rejects_unknown_pdf():
    response = client.post(
        "/ask",
        json={
            "question": "What is this PDF about?",
            "filename": "does-not-exist.pdf",
            "history": [],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Selected PDF not found. Please upload it again."
    )


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": "",
            "filename": "sample.pdf",
            "history": [],
        },
    )

    assert response.status_code == 422
