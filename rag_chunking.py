"""Domain-neutral, structure-preserving document chunker.

Preserves PDF page metadata and DOCX block locations. Markdown table rows are
never cut through a numeric cell; table fragments repeat their headers. Nearby
short DOCX paragraphs may be coalesced without losing their block ranges.
"""
from __future__ import annotations

import re
from typing import Any

from rag_retrieval import extract_headings


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    return _is_table_row(line) and bool(re.fullmatch(r"[| :\-]+", line.strip()))


def _split_words(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    result: list[str] = []
    buf = ""
    for word in re.findall(r"\S+\s*", text):
        if len(word) > limit:
            if buf.strip():
                result.append(buf.strip())
            buf = ""
            result.extend(word[j:j + limit] for j in range(0, len(word), limit))
        elif len(buf) + len(word) > limit:
            result.append(buf.strip())
            buf = word
        else:
            buf += word
    if buf.strip():
        result.append(buf.strip())
    return [t for t in result if t]


def _table_chunks(rows: list[str], heading: str, size: int) -> list[tuple[str, str]]:
    if not rows:
        return []
    headers = rows[:2] if len(rows) >= 2 and _is_table_separator(rows[1]) else rows[:1]
    body = rows[len(headers):]
    # Long tables are split at *row* boundaries. The user can retrieve multiple
    # table fragments instead of receiving an artificially truncated table.
    limit = max(3300, size * 3)
    prefix = (heading[:150] + "\n" if heading else "") + "\n".join(headers)
    results: list[tuple[str, str]] = []
    active: list[str] = []
    for row in body:
        if active and len(prefix) + sum(len(t) + 1 for t in active) + len(row) + 1 > limit:
            results.append((prefix + "\n" + "\n".join(active), "table"))
            active = []
        # A single huge row stays intact. It must never silently lose cells.
        active.append(row)
    results.append((prefix + ("\n" + "\n".join(active) if active else ""), "table"))
    return [(s.strip(), k) for s, k in results if s.strip()]


def _group_docx_blocks(pages: list[dict[str, Any]], chunk_size: int) -> list[dict[str, Any]]:
    """Reduce thousands of tiny DOCX paragraph vectors; preserve block ranges."""
    grouped: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        first, last = buffer[0], buffer[-1]
        if len(buffer) == 1:
            grouped.append(first)
        else:
            grouped.append({**first, 'locator': first['locator'] + '–' + last['locator'],
                            'text': '\n\n'.join(p['text'] for p in buffer),
                            'has_ocr': any(p.get('has_ocr') for p in buffer)})
        buffer = []

    for p in pages:
        text = p.get('text', '').strip()
        is_docx = p.get('kind') == 'docx'
        is_table = bool(re.search(r'^\s*\|.*\|', text, re.M))
        heading = text.startswith('#')
        too_large = len(text) >= chunk_size
        if not is_docx or is_table or heading or too_large:
            flush()
            grouped.append(p)
            continue
        if buffer and (buffer[0].get('doc_id', buffer[0]['filename']) != p.get('doc_id', p['filename'])
                       or sum(len(x['text']) + 2 for x in buffer) + len(text) > chunk_size):
            flush()
        if text:
            buffer.append(p)
    flush()
    return grouped


def chunk_pages(pages: list[dict[str, Any]], chunk_size: int = 1250,
                overlap: int = 180) -> list[dict[str, Any]]:
    if chunk_size < 300 or not (0 <= overlap < chunk_size):
        raise ValueError('Invalid chunk size or overlap')
    chunks: list[dict[str, Any]] = []
    pages = _group_docx_blocks(pages, chunk_size)
    last_heading = ''
    last_doc: str | None = None

    for page in pages:
        text = page.get('text', '').strip()
        if not text:
            continue
        doc_identity = str(page.get('doc_id', page.get('filename')))
        if doc_identity != last_doc:
            last_heading = ''
            last_doc = doc_identity
        meta = {key: page.get(key) for key in ('doc_id', 'filename', 'page', 'locator', 'kind')}
        units: list[tuple[str, str, str]] = []
        paragraph: list[str] = []
        active_heading = last_heading

        def flush_paragraph() -> None:
            nonlocal paragraph
            if not paragraph:
                return
            body = '\n'.join(paragraph).strip()
            paragraph = []
            for sub in re.split(r'\n\s*\n', body):
                if len(sub) <= chunk_size:
                    units.append((sub, 'text', active_heading))
                else:
                    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', sub)
                    merged = ''
                    for part in parts:
                        if merged and len(merged) + 1 + len(part) > chunk_size:
                            units.append((merged.strip(), 'text', active_heading))
                            merged = ''
                        if len(part) > chunk_size:
                            units.extend((v, 'text', active_heading) for v in _split_words(part, chunk_size))
                        else:
                            merged = (merged + ' ' + part).strip()
                    if merged:
                        units.append((merged.strip(), 'text', active_heading))

        lines = text.splitlines()
        detected_headings = {name for name, _, _ in extract_headings(text)}
        i = 0
        while i < len(lines):
            line = lines[i]
            if _is_table_row(line):
                flush_paragraph()
                rows = []
                while i < len(lines) and _is_table_row(lines[i]):
                    rows.append(lines[i].strip())
                    i += 1
                units.extend((body, kind, active_heading)
                             for body, kind in _table_chunks(rows, active_heading, chunk_size))
                continue
            if not line.strip():
                flush_paragraph()
                i += 1
                continue
            if re.match(r'^\s*#{1,6}\s+', line) or line.strip() in detected_headings:
                flush_paragraph()
                active_heading = re.sub(r'^\s*#{1,6}\s+', '', line).strip('*# ')
                last_heading = active_heading
                # Keep heading available as an independent retrievable section.
                paragraph.append(line)
                i += 1
                continue
            paragraph.append(line)
            i += 1
        flush_paragraph()

        buffer: list[str] = []
        buffer_heading = ''

        def emit() -> None:
            nonlocal buffer, buffer_heading
            if not buffer:
                return
            body = '\n\n'.join(buffer).strip()
            prefix = f'[Section: {buffer_heading}]\n' if buffer_heading and buffer_heading.casefold() not in body[:140].casefold() else ''
            chunks.append({**meta, 'text': prefix + body, 'chunk_type': 'text'})
            buffer = []

        for unit, kind, heading in units:
            if not unit.strip():
                continue
            if kind == 'table':
                emit()
                chunks.append({**meta, 'text': unit, 'chunk_type': 'table'})
                continue
            if buffer and (len('\n\n'.join(buffer)) + len(unit) + 2 > chunk_size or heading != buffer_heading):
                before = '\n\n'.join(buffer).strip()
                emit()
                tail = before[-overlap:] if overlap else ''
                if tail and ' ' in tail:
                    tail = re.sub(r'^\S*\s+', '', tail).strip()
                buffer = [tail, unit] if tail and len(tail) + len(unit) < chunk_size else [unit]
            else:
                buffer.append(unit)
            buffer_heading = heading
        emit()
    return chunks
