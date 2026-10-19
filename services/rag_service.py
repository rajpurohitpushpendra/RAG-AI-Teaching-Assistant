
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from app.config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    UPLOAD_DIR,
)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_store():
    return Chroma(
        collection_name="study_materials",
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def process_pdf(filename):
    pdf_path = UPLOAD_DIR / filename

    documents = PyPDFLoader(str(pdf_path)).load()
    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No readable text found. The PDF may be scanned or image-only."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not create text chunks from this PDF.")

    for chunk in chunks:
        chunk.metadata["source"] = filename

    store = get_vector_store()

    existing = store.get(
        where={"source": filename},
        include=["metadatas"],
    )
    existing_ids = existing.get("ids", [])

    if existing_ids:
        store.delete(ids=existing_ids)

    store.add_documents(chunks)

    return len(chunks)


def get_document_chunks(filename):
    store = get_vector_store()

    data = store.get(
        where={"source": filename},
        include=["documents", "metadatas"],
    )

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    chunks = []

    for content, metadata in zip(documents, metadatas):
        if content and content.strip():
            chunks.append({
                "text": content,
                "page": metadata.get("page", 0) + 1,
            })

    return chunks


def answer_question(question, filename, history=None):
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "grounded": False,
        }

    all_chunks = get_document_chunks(filename)

    if not all_chunks:
        return {
            "answer": (
                "I couldn't find indexed text for this PDF. "
                "Please upload the PDF again."
            ),
            "sources": [],
            "grounded": False,
        }

    broad_terms = (
        "about pdf",
        "about my pdf",
        "about this pdf",
        "summarize",
        "summary",
        "overview",
        "tell me about",
        "what is in this",
        "what does this pdf",
        "education and experience",
        "details about education",
        "experience",
        "education",
        "internship details",
        "explain this document",
    )

    normalized_question = question.lower()
    is_broad_question = any(
        term in normalized_question
        for term in broad_terms
    )

    if is_broad_question:
        selected_chunks = all_chunks
    else:
        store = get_vector_store()

        results = store.similarity_search(
            question,
            k=min(6, len(all_chunks)),
            filter={"source": filename},
        )

        selected_chunks = [
            {
                "text": doc.page_content,
                "page": doc.metadata.get("page", 0) + 1,
            }
            for doc in results
            if doc.page_content and doc.page_content.strip()
        ]

        if not selected_chunks:
            selected_chunks = all_chunks[: min(4, len(all_chunks))]

    max_context_chars = 18000
    context_parts = []
    total_chars = 0

    for chunk in selected_chunks:
        part = f"Page {chunk['page']}:\n{chunk['text']}"
        remaining = max_context_chars - total_chars

        if remaining <= 0:
            break

        if len(part) > remaining:
            part = part[:remaining]

        context_parts.append(part)
        total_chars += len(part)

    context = "\n\n".join(context_parts)

    conversation_parts = []

    for item in (history or [])[-6:]:
        role = item.get("role")
        content = item.get("content", "")

        if role in {"user", "assistant"} and content:
            conversation_parts.append(
                f"{role.upper()}: {content[:1000]}"
            )

    conversation = "\n".join(conversation_parts)

    prompt = f"""You are StudyMate AI, a helpful document-based teaching assistant.

IMPORTANT RULES:
1. Answer factual questions using only the selected PDF excerpts below.
2. Never mix information from another PDF.
3. For broad questions about the document, summarize the available excerpts.
4. For education, experience, internships, projects, skills, or certificates,
   extract the relevant details when they appear in the excerpts.
5. If the PDF does not contain the requested information, clearly say so.
6. Do not invent names, dates, qualifications, employers, or achievements.
7. Use recent conversation history only to understand follow-up questions.
   Previous assistant answers are not evidence.
8. Give clear, well-organized answers in simple language.
9. Do not refuse a question merely because it is broad.

Selected PDF: {filename}

Recent conversation:
{conversation or "No previous conversation."}

PDF excerpts:
{context}

Current question:
{question}

Answer based only on the PDF excerpts above:"""

    response = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    ).invoke(prompt)

    source_pages = sorted({
        chunk["page"] for chunk in selected_chunks
    })

    return {
        "answer": response.content,
        "sources": [
            {
                "file": filename,
                "page": page,
            }
            for page in source_pages
        ],
        "grounded": True,
    }
