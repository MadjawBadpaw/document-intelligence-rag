# Document Intelligence RAG — General-Purpose v3

A CLI-based, source-grounded retrieval augmented generation engine for **PDF, DOCX, TXT and Markdown**. It indexes text, preserves Markdown table structure, recognizes headings at runtime, and can OCR text in document images. It does **not** contain syllabus, finance, medicine, legal, or other document-specific retrieval rules.

## What it does

- PDF: layout-aware extraction with PyMuPDF 2.x (`Document.to_markdown`), page metadata, local OCR for scanned pages and selected embedded images. Provides compatibility fallback for older PyMuPDF + `pymupdf4llm` installations.
- DOCX: extracts paragraphs, headings, tables and embedded-image OCR; citations use **block references** because DOCX has no stable physical PDF page numbers.
- TXT / MD: extracts text, headings and Markdown tables.
- Search: `BAAI/bge-base-en-v1.5` embeddings, FAISS, BM25, reciprocal-rank fusion, optional `cross-encoder/ms-marco-MiniLM-L6-v2` reranking.
- Evidence: dynamically matches source-discovered headings and identifiers, keeps relevant complete tables and continuation pages together, and selects diverse evidence rather than blindly relying on reranker scores.
- Answers: generates responses using Groq and references sources as `[S1]`, `[S2]`, etc. The source list includes filename and page/block location.
- Overview: inexpensive *structural/sample-based* overview, usually one Groq request.
- Summary: multi-stage summarization of **all extracted text-bearing pages**. It uses more API calls, supports caching and bounded retry behavior; not guaranteed to fit a free-tier quota.
- Local cache: parsed chunks, embeddings, and completed summaries are cached between runs.

**Important limitations:** No general RAG can guarantee perfect extraction from arbitrary layouts, rotated pages, equations, handwriting, photos, diagrams, or very complex scanned tables. OCR recognizes text, not full diagram semantics. An LLM can still make factual mistakes; verify the cited sources for high-stakes decisions. This is a single-user CLI, not yet a production API, authentication system or frontend.

## Windows CMD installation

Keep your existing `sample.pdf` (or any other document) and your existing `.env`. Extract the ZIP **into the project directory**, replacing only the Python scripts and requirements file. The new version changes its index schema, so the first run will rebuild its cache automatically. Do **not** delete your documents or `.env`.

Activate your virtual environment, then:

```cmd
python -m pip uninstall -y docx
python -m pip install -r requirements-rag.txt
```

The correct Word package is `python-docx`, imported in Python as `from docx import Document`. The unrelated old `docx` package can cause import problems.

Create or keep `.env` in the project root:

```dotenv
GROQ_API_KEY=your_key_here
GROQ_TPM=6000
GROQ_RPM=25
RAG_SUMMARY_BATCH_CHARS=7000
```

Rate parameters are only conservative **starting values**. Your Groq dashboard and individual model quota are authoritative. You may need lower values.

## Commands

**Build or reuse an index and ask questions interactively:**

```cmd
python rag.py report.pdf --ocr auto
python rag.py report.pdf notes.docx --ocr auto
```

Inside the interactive session, type a normal question, `overview`, `summary`, `detailed summary`, `inspect`, or `exit`.

**Inspect extraction and search without using Groq:**

```cmd
python rag.py report.pdf --inspect
python rag.py report.pdf --search "List all project names, owners and due dates"
python rag.py report.pdf --search "What is the escalation procedure?" --no-rerank
```

**Quick overview (usually one API request):**

```cmd
python rag.py report.pdf --overview
```

**All-page summarization (multiple requests, may encounter quotas):**

```cmd
python rag.py report.pdf --summary
python rag.py report.pdf --summary --detailed
```

**OCR settings:** `--ocr off` disables local image OCR; `--ocr auto` OCRs image-only PDF pages and image-containing sections as needed; `--ocr all` additionally attempts to OCR every PDF page, which can be much slower. OCR uses local ONNX models; document text sent to Groq for answer generation is **not** processed by a local LLM.

**Force an index rebuild** only when needed:

```cmd
python rag.py report.pdf --reindex
```

## App-like test client (no actual frontend/backend required yet)

```cmd
python test_rag.py report.pdf
python test_rag.py report.pdf --ask "What are the key findings?"
python test_rag.py report.pdf --ocr auto --top-k 8
python test_rag.py --ocr-smoke
```

Once inside the test client, use `/help`, `/stats`, `/search ...`, `/ask ...`, `/sources`, `/source 1`, `/overview`, `/summary`, and `/export`.

For a user-defined retrieval benchmark, create `checks.json` tailored **to your document**:

```json
[
  {
    "question": "What does the warranty cover?",
    "expected_pages": [2],
    "expected_text": ["warranty"]
  },
  {
    "question": "Where are operational expenses listed?",
    "expected_pages": [19],
    "expected_text": ["expenses"]
  }
]
```

Adapt the expected pages and text to the actual document. Page numbers are **1-based PDF pages**. You may alternatively use `expected_files` for multi-document collections. The benchmark tests **retrieval evidence**, not LLM answer accuracy.

```cmd
python test_rag.py report.pdf --benchmark checks.json
```

## Offline regression tests

```cmd
python test_general_rag.py
```

Tests include general financial tables, large tables, manuals, a 321-page PDF, a DOCX contract, image-only OCR routing, overview/summary caching and a mocked PyMuPDF 2.x extraction route. Mocked encoder, reranker, OCR inference and Groq calls make these tests fast and deterministic. They are **not a substitute** for live-model and real-OCR evaluation on your Windows installation.

Optional regression against the previously used university PDF, strictly as a *test fixture*:

```cmd
python test_general_rag.py --syllabus "CSE (3).pdf"
```

The engine itself contains no fixed syllabus headings, course codes or page positions.

## Source layout

| File | Role |
|---|---|
| `rag.py` | Public CLI and `DocumentRAG` orchestration: parsing, OCR, index, generation, summaries, cache. |
| `rag_retrieval.py` | General-purpose query planning, hybrid retrieval, dynamic headings, evidence selection. |
| `rag_chunking.py` | Table-aware and structure-aware chunking. |
| `test_rag.py` | Application-style interactive test client, JSON report and custom benchmark. |
| `test_general_rag.py` | Offline regression suite. |
| `requirements-rag.txt` | Dependency specifications. |
| `.env.example` | Safe configuration template (no actual secrets). |

## Before building FastAPI / the frontend

Validate (1) local retrieval with `--search`, (2) cited answers for concrete questions via `--ask`, (3) OCR on a real scanned document, (4) a short `--overview`, and only then (5) the expensive full `--summary`. The future backend can call `DocumentRAG` directly instead of duplicating its logic.

## Privacy and quotas

Indexing, embeddings, and OCR run locally. Generating answers and summaries uses the configured **Groq-hosted model**; selected source excerpts or summary batches are sent to Groq. Do not process confidential documents without reviewing applicable data-handling requirements. Never commit `.env` or API keys.
