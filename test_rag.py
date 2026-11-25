"""Interactive integration test client for the current DocumentRAG engine.

This deliberately emulates a future frontend calling a backend service:
    UPLOAD -> STATUS -> SEARCH/ASK -> CITATIONS -> SUMMARY -> EXPORT.
It DOES NOT modify rag.py and never embeds/duplicates the retrieval algorithm.

Examples (Windows CMD, from project directory):
    python test_rag.py handbook.pdf
    python test_rag.py handbook.pdf --ocr auto --top-k 8
    python test_rag.py handbook.pdf --benchmark checks.json
    python test_rag.py handbook.pdf --ask "What does the document say about deployment?"
    python test_rag.py --ocr-smoke

During a session type /help for commands. The first embedding/reranker/OCR
model downloads may take time. Ordinary retrieval makes no Groq API call.
Full-document summaries make MANY Groq calls; the client asks for confirmation.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag import DocumentRAG, _is_summary, _is_overview

SUPPORTED = {'.pdf', '.docx', '.txt', '.md'}


def _source_dict(raw: dict[str, Any], position: int) -> dict[str, Any]:
    """Single normalized evidence contract suitable for a future API response."""
    score = raw.get('score')
    rerank = raw.get('rerank_score')
    return {
        'id': f'S{position}',
        'filename': str(raw.get('filename', '')),
        'page': raw.get('page'),  # For PDFs: 1-based; DOCX has no physical page number.
        'locator': str(raw.get('locator', '')),
        'kind': str(raw.get('kind', '')),
        'similarity': float(score) if score is not None else None,
        'rerank_score': float(rerank) if rerank is not None else None,
        'text': str(raw.get('text', '')),
        'evidence_type': raw.get('evidence_type', 'chunk'),
        'reason': raw.get('reason', ''),
    }


def _print_sources(sources: list[dict[str, Any]], preview_chars: int = 240) -> None:
    if not sources:
        print('  No retrieval passages (a whole-document summary does not return top-k citations).')
        return
    for item in sources:
        loc = f'page {item["page"]}' if item['page'] else item['locator']
        sim = item['similarity']
        score = f' | cosine={sim:.3f}' if sim is not None else ''
        rr = item['rerank_score']
        rank = f' | rerank={rr:.3f}' if rr is not None else ''
        print(f'  [{item["id"]}] {item["filename"]} ({loc}){score}{rank} | {item.get("evidence_type", "chunk")}')
        if preview_chars:
            excerpt = ' '.join(item['text'].split())
            print('       ' + excerpt[:preview_chars] + ('…' if len(excerpt) > preview_chars else ''))


def _short_stats(stats: dict[str, Any]) -> None:
    print('\nINDEX STATUS')
    for key in ('documents', 'pdf_pages', 'pdf_pages_with_text', 'empty_pdf_pages',
                'docx_blocks', 'ocr_added_blocks', 'chunks'):
        print(f'  {key:23} {stats.get(key, 0)}')
    failures = stats.get('ocr_failures') or []
    if failures:
        print(f'  OCR failures: {len(failures)} (use /ocr-errors to inspect)')
    else:
        print('  OCR failures: 0')
    print('  Note: 0 OCR-added blocks is normal for PDFs that already contain selectable text.')


class RAGTestClient:
    """Thin application boundary that a future API can reuse or adapt."""

    def __init__(self, *, ocr: str = 'auto', rerank: bool = True,
                 top_k: int = 8, device: str | None = None):
        self.engine = DocumentRAG(ocr=ocr, rerank=rerank, device=device)
        self.top_k = top_k
        self.files: dict[str, tuple[Path, bytes]] = {}
        self.history: list[dict[str, Any]] = []
        self.last_sources: list[dict[str, Any]] = []

    def upload(self, paths: list[Path], *, append: bool = False,
               reindex: bool = False) -> dict[str, Any]:
        """Simulates POST /documents, using the SAME input contract as rag.py."""
        next_files = dict(self.files) if append else {}
        for supplied in paths:
            path = supplied.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'File not found: {path}')
            if path.suffix.lower() not in SUPPORTED:
                raise ValueError(f'Unsupported file: {path.name} (use PDF, DOCX, TXT, MD)')
            if path.name in next_files and next_files[path.name][0] != path:
                raise ValueError(f'Duplicate filename: {path.name}. Rename one file to avoid ambiguous citations.')
            next_files[path.name] = (path, path.read_bytes())
        if not next_files:
            raise ValueError('Upload at least one file')

        # Only replace the active document set after successful indexing.
        documents = [(name, data) for name, (_, data) in next_files.items()]
        started = time.perf_counter()
        count = self.engine.build_index(documents, refresh=reindex)
        self.files = next_files
        self.last_sources = []
        self.history.clear()  # Questions from an earlier document set do not remain attached.
        return {
            'status': 'ready', 'files': list(self.files), 'chunks': count,
            'stats': self.engine.stats(),
            'elapsed_seconds': round(time.perf_counter() - started, 2),
        }

    def status(self) -> dict[str, Any]:
        """Simulates GET /documents/status."""
        return {'status': 'ready' if self.engine.index is not None else 'empty',
                'files': list(self.files), 'stats': self.engine.stats()}

    def search(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        """Simulates POST /search; retrieval-only, zero Groq tokens."""
        started = time.perf_counter()
        raw = self.engine.retrieve(question, top_k=top_k or self.top_k)
        evidence = [_source_dict(hit, i) for i, hit in enumerate(raw, 1)]
        self.last_sources = evidence
        return {'type': 'search', 'query': question, 'sources': evidence,
                'elapsed_seconds': round(time.perf_counter() - started, 2)}

    def ask(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        """Simulates POST /ask with structured answer + source metadata."""
        started = time.perf_counter()
        raw = self.engine.ask(question, top_k=top_k or self.top_k)
        evidence = [_source_dict(hit, i) for i, hit in enumerate(raw.get('sources') or [], 1)]
        answer = str(raw['answer'])
        cited_ids = sorted(set(re.findall(r'\[S(\d+)\]', answer)), key=int)
        bad = [f'S{number}' for number in cited_ids
               if int(number) > len(evidence) or int(number) < 1]
        result = {
            'type': 'answer', 'question': question,
            'mode': raw.get('mode', 'retrieval'),
            'answer': answer, 'sources': evidence,
            'citation_ids': [f'S{number}' for number in cited_ids],
            'invalid_citation_ids': bad,
            'elapsed_seconds': round(time.perf_counter() - started, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        self.last_sources = evidence
        self.history.append(result)
        return result

    def overview(self) -> dict[str, Any]:
        """One Groq call from section/structure evidence; not exhaustive."""
        started = time.perf_counter()
        answer = self.engine.overview_document()
        result = {'type': 'overview', 'mode': 'document-overview', 'answer': answer,
                  'sources': [], 'coverage': self.engine.stats(),
                  'elapsed_seconds': round(time.perf_counter() - started, 2),
                  'timestamp': datetime.now(timezone.utc).isoformat()}
        self.last_sources = []
        self.history.append(result)
        return result

    def summarize(self, *, detailed: bool = False) -> dict[str, Any]:
        """Simulates POST /summaries; reads every text-bearing page (expensive)."""
        started = time.perf_counter()
        answer = self.engine.summarize_document('detailed' if detailed else 'standard')
        result = {
            'type': 'summary', 'mode': 'document-summary',
            'detail': 'detailed' if detailed else 'standard',
            'answer': answer, 'sources': [],
            'coverage': self.engine.stats(),
            'elapsed_seconds': round(time.perf_counter() - started, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        self.last_sources = []
        self.history.append(result)
        return result

    def save_report(self, filename: Path) -> Path:
        """Save app-like JSON results; never includes the GROQ_API_KEY."""
        report = {'generated_at': datetime.now(timezone.utc).isoformat(),
                  'backend': 'DocumentRAG', 'files': list(self.files),
                  'stats': self.engine.stats(), 'history': self.history}
        filename = filename.expanduser().resolve()
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        return filename


def run_benchmark(client: RAGTestClient, path: Path) -> bool:
    """User-defined retrieval benchmark: no document-specific assumptions.

    Input JSON is an array of cases:
      [{"question":"...", "expected_pages":[3], "expected_text":["invoice total"],
        "expected_files":["report.pdf"]}]
    A case passes if all provided expectations appear somewhere in top-k sources.
    An empty expected_* field is skipped. This does NOT test LLM faithfulness.
    """
    cases = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Benchmark must be a nonempty JSON array of test cases')
    passed = 0
    mrr = []
    for idx, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not str(case.get('question', '')).strip():
            raise ValueError(f'Benchmark case {idx} requires a question')
        items = client.search(str(case['question']))['sources']
        expected_pages = {int(n) for n in case.get('expected_pages', [])}
        expected_files = {str(n) for n in case.get('expected_files', [])}
        expected_text = [str(v).casefold() for v in case.get('expected_text', [])]
        found_pages = {s['page'] for s in items}
        found_files = {s['filename'] for s in items}
        joined_text = '\n'.join(s['text'].casefold() for s in items)
        okay = (expected_pages <= found_pages and expected_files <= found_files
                and all(term in joined_text for term in expected_text))
        passed += okay
        relevant_rank = next((j for j, s in enumerate(items, 1)
                              if ((not expected_pages or s['page'] in expected_pages)
                                  and (not expected_files or s['filename'] in expected_files)
                                  and (not expected_text or any(t in s['text'].casefold() for t in expected_text)))), None)
        mrr.append(1.0 / relevant_rank if relevant_rank is not None else 0.0)
        print(f'  {idx:2}. {"PASS" if okay else "FAIL"} | {case["question"][:95]}')
    print(f'PASS {passed}/{len(cases)} | Recall@{client.top_k}: {passed/len(cases):.1%} '
          f'| MRR: {sum(mrr)/len(mrr):.3f}')
    print('Retrieval benchmark only: inspect Groq answers separately for factual accuracy.')
    return passed == len(cases)


def run_ocr_smoke() -> bool:
    """Create a raster-only PDF and verify that the backend can OCR its image.

    Skipped unless explicitly requested: OCR runtimes/models may need downloads.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pymupdf
    except ImportError as exc:
        print(f'OCR smoke test needs Pillow and PyMuPDF: {exc}')
        return False
    img = Image.new('RGB', (1400, 340), 'white')
    draw = ImageDraw.Draw(img)
    font = None
    for candidate in ('C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try:
            font = ImageFont.truetype(candidate, 55)
            break
        except (OSError, ValueError):
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((45, 65), 'NETWORK SECURITY TEST', font=font, fill='black')
    draw.text((45, 155), 'PROJECT CODE 4827', font=font, fill='black')
    stream = io.BytesIO()
    img.save(stream, format='PNG')
    pdf = pymupdf.open()
    try:
        page = pdf.new_page(width=720, height=210)
        page.insert_image(pymupdf.Rect(0, 0, 720, 210), stream=stream.getvalue())
        data = pdf.tobytes()
    finally:
        pdf.close()
    test = DocumentRAG(ocr='auto', rerank=False)
    pages = test.extract_pdf(data, 'image_only_test.pdf')
    extracted = ' '.join(p.get('text', '') for p in pages)
    ok = bool(extracted.strip()) and ('NETWORK' in extracted.upper() or '4827' in extracted)
    print('\nOCR SMOKE TEST')
    print('  Native text in synthetic PDF: none (raster image only)')
    print(f'  OCR extracted: {extracted[:400]!r}')
    print(f'  Result: {"PASS" if ok else "FAIL"}')
    if test.ocr_failures:
        for failure in test.ocr_failures[:3]:
            print(f'  OCR error: {failure}')
    return ok


def _help() -> None:
    print('''
COMMANDS (emulating frontend -> backend requests)
  /upload FILE [FILE ...]     Replace the document library and index it
  /add FILE [FILE ...]        Add document(s), then rebuild/reuse the index
  /stats                      Show coverage and indexing statistics
  /search QUESTION            Hybrid search + reranking, NO Groq API calls
  /ask QUESTION               Generate a grounded answer with source IDs
  QUESTION                    Shortcut for /ask QUESTION
  /overview                   Quick document overview (normally ONE Groq call)
  /summary                    Exhaustive document map-reduce (many Groq calls)
  /detailed                   Detailed full-document overview (many calls)
  /sources                    Show last answer's retrieved evidence
  /source NUMBER              Show full text of evidence S1, S2, ...
  /benchmark FILE.json        Run your own retrieval benchmark cases (no Groq)
  /ocr-errors                 List OCR errors, if any
  /history                    Show questions asked in this session
  /export FILE.json           Export questions, answers, evidence and stats
  /help                       Show this help
  /exit                       Quit

NOTE: PDF pages are 1-based. DOCX uses block IDs because DOCX has no fixed pages.
      Cosine similarity and rerank scores are NOT calibrated correctness probabilities.
''')


def interactive(client: RAGTestClient, *, auto_confirm_summary: bool = False) -> None:
    _help()
    while True:
        try:
            line = input('\nRAG> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nClosing test client.')
            return
        if not line:
            continue
        command, _, content = line.partition(' ')
        cmd = command.lower()
        try:
            if cmd in {'/exit', '/quit', '/q', 'exit', 'quit'}:
                print('Closing test client.')
                return
            if cmd == '/help':
                _help()
            elif cmd in {'/upload', '/add'}:
                if not content.strip():
                    print('Usage: /upload "report.pdf" notes.docx')
                    continue
                # posix=False preserves Windows backslashes; remove surrounding quotes.
                files = [Path(p.strip('\"\'')) for p in shlex.split(content, posix=False)]
                print('Indexing documents ...')
                event = client.upload(files, append=(cmd == '/add'))
                print(f'READY: {len(event["files"])} file(s); {event["chunks"]} chunks '
                      f'in {event["elapsed_seconds"]} seconds.')
                _short_stats(event['stats'])
            elif cmd == '/stats':
                _short_stats(client.status()['stats'])
            elif cmd == '/search':
                if not content.strip():
                    print('Usage: /search What are the warranty terms?')
                    continue
                result = client.search(content)
                print(f'RETRIEVAL ({result["elapsed_seconds"]:.2f}s; Groq calls: 0)')
                _print_sources(result['sources'], preview_chars=360)
            elif cmd in {'/ask'} or not cmd.startswith('/'):
                question = content if cmd == '/ask' else line
                if not question.strip():
                    print('Usage: /ask YOUR QUESTION')
                    continue
                if _is_summary(question) and not auto_confirm_summary:
                    answer = input('Full-document summarization may make many API calls. Proceed? [y/N] ')
                    if answer.strip().casefold() not in {'y', 'yes'}:
                        print('Cancelled. Use a targeted question or /search instead.')
                        continue
                result = client.ask(question)
                print(f'\nANSWER (mode={result["mode"]}, {result["elapsed_seconds"]:.2f}s)\n')
                print(result['answer'])
                print('\nRETRIEVED EVIDENCE:')
                _print_sources(result['sources'])
                if result['invalid_citation_ids']:
                    print('WARNING: unrecognized citation IDs: ' + ', '.join(result['invalid_citation_ids']))
                if result['mode'] == 'retrieval' and not result['citation_ids']:
                    print('WARNING: The answer contains no [S#] source citations.')
            elif cmd == '/overview':
                result = client.overview()
                print(f'\nQUICK OVERVIEW ({result["elapsed_seconds"]:.2f}s)\n\n' + result['answer'])
            elif cmd in {'/summary', '/detailed'}:
                if not auto_confirm_summary:
                    answer = input('This summarizes ALL pages and may use many Groq calls. Proceed? [y/N] ')
                    if answer.strip().casefold() not in {'y', 'yes'}:
                        print('Cancelled.')
                        continue
                result = client.summarize(detailed=(cmd == '/detailed'))
                print(f'\nDOCUMENT SUMMARY ({result["elapsed_seconds"]:.2f}s)\n')
                print(result['answer'])
            elif cmd == '/sources':
                _print_sources(client.last_sources, preview_chars=500)
            elif cmd == '/source':
                try:
                    number = int(content.strip().removeprefix('S').removeprefix('s'))
                except ValueError:
                    print('Usage: /source 1')
                    continue
                if not (1 <= number <= len(client.last_sources)):
                    print(f'No S{number}; available: 1 to {len(client.last_sources)}')
                    continue
                source = client.last_sources[number - 1]
                print(f'\n[S{number}] {source["filename"]} ({source["locator"]})\n')
                print(source['text'])
            elif cmd == '/benchmark':
                if not content.strip():
                    print('Usage: /benchmark checks.json')
                else:
                    run_benchmark(client, Path(content.strip().strip('"\'')))
            elif cmd == '/ocr-errors':
                failures = client.engine.stats().get('ocr_failures', [])
                print('\n'.join(failures) if failures else 'No recorded OCR errors.')
            elif cmd == '/history':
                for i, entry in enumerate(client.history, 1):
                    prompt = entry.get('question', entry.get('detail', 'summary'))
                    print(f'{i:2}. [{entry["type"]}] {prompt}')
                if not client.history:
                    print('No answer-generation requests yet.')
            elif cmd == '/export':
                if not content.strip():
                    print('Usage: /export rag-test-report.json')
                    continue
                path = client.save_report(Path(content.strip().strip('\"\'')))
                print(f'Saved test report: {path}')
                print('Note: this report contains question text and retrieved document passages.')
            else:
                print('Unknown command. Use /help.')
        except Exception as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
            print('The client is still active; correct the input and retry.')


def main() -> int:
    parser = argparse.ArgumentParser(description='Integration test harness for the existing rag.py')
    parser.add_argument('files', metavar='FILE', nargs='*', type=Path,
                        help='PDF/DOCX/TXT/MD file paths, e.g. "report.pdf"')
    parser.add_argument('--ocr', choices=['off', 'auto', 'all'], default='auto')
    parser.add_argument('--no-rerank', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--reindex', action='store_true')
    parser.add_argument('--benchmark', metavar='CASES.json', type=Path,
                        help='Run custom document-neutral retrieval benchmark and exit')
    parser.add_argument('--ask', metavar='QUESTION', help='Ask one question and exit')
    parser.add_argument('--ocr-smoke', action='store_true',
                        help='Test OCR on a generated image-only PDF (does not need a document)')
    parser.add_argument('--yes-summary', action='store_true',
                        help='Skip the confirmation before costly document-wide summaries')
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 40:
        parser.error('--top-k must be between 1 and 40')
    if args.ocr_smoke:
        return 0 if run_ocr_smoke() else 1

    client = RAGTestClient(ocr=args.ocr, rerank=not args.no_rerank,
                           top_k=args.top_k, device=args.device)
    if args.files:
        try:
            print('UPLOAD -> PARSE -> OCR (if needed) -> CHUNK -> EMBED -> INDEX')
            result = client.upload(args.files, reindex=args.reindex)
        except Exception as exc:
            print(f'Indexing failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
        print(f'READY: {result["chunks"]} chunks indexed from '
              f'{len(result["files"])} file(s) in {result["elapsed_seconds"]}s.')
        _short_stats(result['stats'])
    elif args.benchmark or args.ask:
        parser.error('Provide at least one document for --benchmark or --ask')
    else:
        print('No documents loaded yet. Type /upload "myfile.pdf" to begin.')

    if args.benchmark:
        return 0 if run_benchmark(client, args.benchmark) else 1
    if args.ask:
        if _is_summary(args.ask) and not args.yes_summary:
            parser.error('This is a whole-document summary request; add --yes-summary to permit many API calls')
        try:
            result = client.ask(args.ask)
        except Exception as exc:
            print(f'Question failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
        print('\nANSWER:\n' + result['answer'])
        print('\nSOURCES:')
        _print_sources(result['sources'])
        return 0

    interactive(client, auto_confirm_summary=args.yes_summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
