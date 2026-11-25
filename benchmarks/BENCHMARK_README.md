# Document Intelligence RAG — Evaluation Benchmark v1

**50 labeled questions:** 20 real university syllabus PDF, 20 real Infosys annual report PDF, and 10 deterministic cross-format fixture questions (DOCX, Markdown, TXT, scanned PDF). Four are deliberately unanswerable. Cases are organized into 38 development and 12 holdout questions.

This benchmark **does not modify `rag.py`, `rag_retrieval.py`, or your existing index**. It calls your actual `DocumentRAG.retrieve()` and (in full mode) `DocumentRAG.ask()` APIs. It does not assume a correct result: it records misses and failures.

## Installation in your existing project

Keep this structure (the benchmark files are provided separately):

```text
document-intelligence-rag/
├── rag.py
├── rag_retrieval.py
├── rag_chunking.py
├── .env                  # Private GROQ_API_KEY, RAG_DEVICE=cuda, etc.
├── sample.pdf            # Existing 160-page university syllabus
├── report.pdf            # Existing Infosys 2024-25 integrated annual report
└── benchmarks/
    ├── run_benchmark.py
    ├── benchmark_cases.json
    ├── benchmark_fixtures.py
    └── BENCHMARK_README.md
```

Run in **CMD with your existing `.venv` active**, from the project root:

```cmd
python benchmarks\benchmark_fixtures.py --out-dir .
python benchmarks\run_benchmark.py --data-dir . --self-test
```

The first command generates four synthetic files in the project root: `soc_runbook.docx`, `api_reference.md`, `release_notes.txt`, and `scanned_invoice.pdf`. They are **fictional test fixtures**, not outside-grounded documents. You need the `python-docx` and `Pillow` packages already used by the RAG stack.

## 1. Retrieval benchmark (no Groq calls)

```cmd
python benchmarks\run_benchmark.py --data-dir . --mode retrieval --split dev --out benchmark_retrieval_dev.json
```

This evaluates **38 development questions** using the real BGE/FAISS/BM25/CrossEncoder pipeline (unless you use `--no-rerank`). With `--ocr auto` (default), the scanned invoice is OCR'd in the fixtures corpus. It reuses `.rag_cache` when the source/index configuration matches, so no forced reindexing is needed.

After code has stabilized, evaluate the 12 questions withheld from tuning:

```cmd
python benchmarks\run_benchmark.py --data-dir . --mode retrieval --split holdout --out benchmark_retrieval_holdout.json
```

If you want all 50 in one report:

```cmd
python benchmarks\run_benchmark.py --data-dir . --mode retrieval --out benchmark_retrieval_all.json
```

To restrict testing during debugging:

```cmd
python benchmarks\run_benchmark.py --data-dir . --corpus syllabus --limit 5
python benchmarks\run_benchmark.py --data-dir . --corpus report --split dev
python benchmarks\run_benchmark.py --data-dir . --corpus fixtures --only-id X09
```

To compare performance **with and without reranking**, run the same corpus/split twice with different `--out` filenames; add `--no-rerank` to the second command. The result is an ablation, not a pure dense-only baseline.

## 2. Full-answer benchmark (uses your Groq key)

Start small, because 50 sequential Groq requests may run into free-tier token/rate limits:

```cmd
python benchmarks\run_benchmark.py --data-dir . --mode full --corpus report --limit 5 --out benchmark_answers_smoke.json
```

Once ready:

```cmd
python benchmarks\run_benchmark.py --data-dir . --mode full --split holdout --out benchmark_answers_holdout.json
```

This creates **three files**: `benchmark_answers_holdout.json`, `benchmark_answers_holdout.md` (readable report), and `benchmark_answers_holdout.review.csv`.

Review the CSV. For every response, manually enter **1** (correct) or **0** (incorrect) in `answer_correct` and `citations_correct`, and add a note about failures. A correct-looking answer without supporting citations should be marked incorrect for citation correctness.

Recompute **human-validated accuracy** without calling Groq again:

```cmd
python benchmarks\run_benchmark.py --rescore-report benchmark_answers_holdout.json --judgments benchmark_answers_holdout.review.csv
```

That saves `benchmark_answers_holdout.graded.json` and a human-graded Markdown report.

## Metric definitions and limitations

| Metric | Definition | Important caveat |
|---|---|---|
| **Evidence Recall@5** | Fraction of required, manually annotated evidence *groups* found in the first 5 returned source excerpts | Gold groups have OR alternatives. This is **not** exhaustive chunk-level recall. |
| **Precision@5 (fixed)** | Number of first 5 excerpts matching any gold group, divided by 5 | A valid RAG returning only 1 perfect citation scores 0.2; inspect purity too. |
| **Source Purity@5** | Relevant returned excerpts divided by the actual number returned (up to 5) | More suitable when retriever intentionally returns fewer than 5 sources. |
| **Citation ID validity** | Cited `[S1]` / `【S1】` IDs exist in the returned source list | Does not prove each claim is supported. |
| **Citation alignment** | Cited source excerpt overlaps any annotated gold evidence group | Passage-level proxy, **not** claim-by-claim correctness. |
| **Citation gold coverage** | Required evidence groups supported by at least one cited source | A good source can support several required groups. |
| **Automatic answer-key coverage** | Number/phrase groups from the answer key found in generated answer | Lexical coverage is **not** semantic correctness. It may miss paraphrases or accept misattributions. |
| **Human answer accuracy** | Manually reviewed 1/0 correctness from `review.csv` | Report separately as the credible final accuracy measure. |
| **Abstention rate** | Percentage of 4 unanswerable cases that explicitly reject an unsupported answer | Language heuristic; review manually. |

**No metric is automatically trusted to detect factual hallucination**. Review wrong units, mislabeled columns, calculations, missing footnotes, and claim-level citation support. Source text matching uses punctuation/case-insensitive spans and may still be imperfect. All questions and labels are visible; the holdout is a workflow discipline, not a cryptographic blind evaluation.

### Benchmark design cautions

- **Two actual PDFs, four synthetic test formats:** 40 real-document cases offer real table and academic data, but this does not establish generalization to every document domain. Add additional third-party manuals, invoices and contracts over time.
- **Many annual-report questions share page 30.** Per-corpus scores and the dedicated page-196/276/321 source-specific cases are important; do not claim 50 independent documents or use the scores as proof of universal performance.
- **The scanned invoice** tests OCR against a known raster image. If it is missed, that's useful evidence of an OCR or extraction failure. Do not silently exclude X09.
- **Four abstention cases** have no gold relevant passage; recall and precision are undefined for these, and are excluded from their denominators. Source hallucination and abstention are evaluated separately.
- **`--ocr auto` can rebuild caches** when OCR policy changes; otherwise index caching is automatic.
- **Review first** before placing any percentage in a resume or README. Keep the exact question set, code commit, model configuration, and benchmark output alongside every performance claim.

The Infosys annual report gold values were checked against the official 2024-25 report: https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-25.pdf. The university PDF's markers were checked against the uploaded 160-page document. The financial PDF was not available for a full programmatic end-to-end run in this authoring environment; verify the page labels on your local copy when executing the benchmark. Four fixture files are synthetic and deterministic.
