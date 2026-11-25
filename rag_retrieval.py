"""Domain-neutral hybrid retrieval and evidence assembly.

No industry, syllabus, subject, document title, page number or course ID is
built into this module. Heading matches are discovered *from the uploaded
text*. Exact identifiers are extracted from each question at runtime.

Pipeline: dense + sparse retrieval -> RRF -> cross-encoder -> diversified
source selection; independently preserve high-confidence page/section/table
anchors when the question requires a complete set of structured facts.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rag import DocumentRAG

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)
_PAGE_RE = re.compile(r"\bpage\s*(?:number|no\.?|#)?\s*[:.]?\s*(\d{1,6})\b", re.I)
_IDENT_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+){1,5}(?![A-Za-z0-9])")
_TABLE_ROW_RE = re.compile(r"^\s*\|[^\n]*\|\s*$")
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s*(.*?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s*\*\*(.*?)\*\*\s*$")
_STOP = set("a an and are as at be by can could did do does for from give has have how i in is it its me my of on or our please show that the their there these this to us was were what when where which who why will with you your pdf doc document file page list all every each any tell provide identify find give details about".split())
_LIST_RE = re.compile(r"\b(list|enumerat\w*|all|every|each|table|tables|compare|comparison|breakdown|total|how many|which are|what are the|names and|codes and|values and)\b", re.I)
_TABLE_RE = re.compile(r"\b(table|tables|rows?|columns?|list|all|every|each|values?|figures?|amounts?|codes?|totals?|compare|breakdown|enumerate|credits?|costs?|prices?|dates?)\b", re.I)
_LONG_DETAIL_RE = re.compile(r"\b(explain|describe|discuss|analy[sz]e|how does|why does|detailed|steps|methodology)\b", re.I)
_ROMANS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
           "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
           "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20}


def tokens(text: str) -> list[str]:
    """Normalize whitespace, punctuation and small Roman section numerals."""
    found = re.findall(r"[A-Za-z0-9]+", text)
    return [str(_ROMANS[t.upper()]) if t.upper() in _ROMANS and len(t) > 1 else t.lower()
            for t in found]


def _content_terms(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _STOP and (len(t) > 1 or t.isdigit())}


def _ids(question: str) -> tuple[str, ...]:
    found: list[str] = []
    for m in _IDENT_RE.finditer(question):
        candidate = m.group()
        if any(c.isdigit() for c in candidate) and any(c.isalpha() for c in candidate) and len(candidate) >= 4:
            found.append(candidate.upper())
    return tuple(dict.fromkeys(found))[:8]


def _normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip(" *#\t"))


def extract_headings(text: str) -> list[tuple[str, int, int]]:
    """Return (heading, start offset, end offset) discovered from Markdown/plain lines."""
    out: list[tuple[str, int, int]] = []
    offset = 0
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        match = _MARKDOWN_HEADING.match(stripped) or _BOLD_HEADING.match(stripped)
        if match:
            heading = _normalize_line(match.group(1))
        elif (stripped and len(stripped) <= 95 and not stripped.startswith("|")
              and not stripped.endswith((".", ":", ",", ";"))
              and not re.search(r"\S\s{2,}\S", stripped)
              and len(stripped.split()) <= 11 and not stripped.startswith(("-", "•"))):
            # Native PDF text sometimes has no Markdown heading markup. Restrict
            # plain-text candidates to visually plausible headings (uppercase,
            # numbered or isolated short lines) rather than treating prose as headings.
            alpha = [c for c in stripped if c.isalpha()]
            capital_ratio = sum(c.isupper() for c in alpha) / max(1, len(alpha))
            numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*|[IVX]{1,5})[.\-:]?\s+", stripped))
            short_isolated = (i + 1 < len(lines) and not lines[i + 1].strip()
                              and len(stripped.split()) <= 7)
            # Titles immediately preceding a Markdown table are a useful
            # heading cue in converted PDFs and simple TXT/MD documents.
            table_follows = (i + 1 < len(lines) and lines[i + 1].lstrip().startswith('|'))
            words = stripped.split()
            title_case = (len(words) <= 10 and len(words) >= 2 and
                          sum(bool(w[:1].isupper()) for w in words) >= len(words) * 0.67)
            first_line_title = (i == 0 and title_case and len(stripped) <= 80)
            heading = stripped if (numbered or capital_ratio > 0.70 or short_isolated
                                   or table_follows or first_line_title) else ""
        else:
            heading = ""
        if heading and (len(heading) >= 4 or match is not None) and len(heading) <= 95:
            out.append((heading, offset, offset + len(line)))
        offset += len(line)
    return out


def _heading_match(heading: str, query_words: list[str]) -> float:
    """A discovered heading must be named by the query; no document-topic rules."""
    words = tokens(heading)
    if not words or len(words) > 12:
        return 0.0
    meaningful = [w for w in words if w not in _STOP]
    if not meaningful:
        return 0.0
    # All heading tokens, in order, must occur in the user query for a strong
    # match. E.g. 'Section 4' must not match 'Section 40'.
    if len(words) <= len(query_words):
        for i in range(len(query_words) - len(words) + 1):
            if query_words[i:i + len(words)] == words:
                return 1.0
    # Allow a query naming the first part of a longer heading, like 'Chapter 4'
    # matching 'Chapter 4: Network Architecture'. Require at least two tokens.
    max_prefix = min(len(words), len(query_words), 5)
    for size in range(max_prefix, 1, -1):
        prefix = words[:size]
        if sum(w not in _STOP for w in prefix) < 1:
            continue
        for i in range(len(query_words) - size + 1):
            if query_words[i:i + size] == prefix:
                return min(0.86, size / max(2, len(words)))
    return 0.0


@dataclass(frozen=True)
class QueryPlan:
    intent: str = "general"
    page_number: int | None = None
    identifiers: tuple[str, ...] = ()
    wants_table: bool = False
    wants_list: bool = False


def plan_query(question: str) -> QueryPlan:
    page = _PAGE_RE.search(question)
    wants_list = bool(_LIST_RE.search(question))
    wants_table = bool(_TABLE_RE.search(question))
    intent = ("page" if page else "enumeration" if wants_list else
              "explanation" if _LONG_DETAIL_RE.search(question) else "general")
    return QueryPlan(intent=intent, page_number=int(page.group(1)) if page else None,
                     identifiers=_ids(question), wants_table=wants_table, wants_list=wants_list)


def _page_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("doc_id", item.get("filename", ""))), str(item.get("locator", "")))


def _has_table(text: str) -> bool:
    """Recognize Markdown tables and aligned plaintext tables (PDF fallback)."""
    if sum(bool(_TABLE_ROW_RE.match(s)) for s in text.splitlines()) >= 3:
        return True
    # Plain PyMuPDF extraction often uses multiple spaces to preserve visual
    # columns. Do not demand any particular header or field names.
    aligned = 0
    for row in text.splitlines():
        if len(row) > 20 and re.search(r'\S\s{3,}\S', row) and len(row.split()) >= 3:
            aligned += 1
    return aligned >= 5


def _table_header_terms(text: str) -> set[str]:
    """Get actual Markdown column-header words, with no domain-specific names."""
    lines = text.splitlines()
    result: set[str] = set()
    for i in range(len(lines) - 1):
        if (_TABLE_ROW_RE.match(lines[i]) and
            re.fullmatch(r'[| :\-]+', lines[i + 1].strip())):
            for term in _content_terms(lines[i]):
                result.add(term.removesuffix('s') if len(term) > 4 else term)
    return result


def _page_candidates(pages: list[dict[str, Any]], question: str, plan: QueryPlan) -> list[tuple[float, int, dict[str, Any], str, str | None]]:
    question_words = tokens(question)
    qterms = _content_terms(question)
    scored = []
    for i, page in enumerate(pages):
        text = page.get("text", "")
        if not text.strip():
            continue
        headings = extract_headings(text)
        best_heading, match_strength = None, 0.0
        for heading, _, _ in headings:
            strength = _heading_match(heading, question_words)
            if strength > match_strength:
                match_strength, best_heading = strength, heading
        ids_found = [code for code in plan.identifiers if re.search(r"(?<!\w)" + re.escape(code) + r"(?!\w)", text, re.I)]
        is_table = _has_table(text)
        # Heading match is especially useful for an enumeration because the
        # complete source table may be scored poorly by a CrossEncoder.
        score = match_strength * 22 + len(ids_found) * 24
        why = []
        if match_strength:
            why.append("matching document heading")
        if ids_found:
            why.append("matching identifier(s): " + ", ".join(ids_found))
        if plan.page_number is not None and page.get("page") == plan.page_number:
            score += 90
            why.append("explicit page number")
        if is_table and match_strength >= 0.5 and plan.wants_table:
            score += 14
            why.append("relevant structured table")
        if is_table and ids_found and plan.wants_table:
            score += 7
        if is_table and plan.wants_table and not (match_strength or ids_found):
            # A query like 'list product names and prices' may not mention any
            # section title or record ID; match the *discovered* column labels.
            header_terms = _table_header_terms(text)
            normalized_qterms = {w.removesuffix('s') if len(w) > 4 else w for w in qterms}
            overlap = len(header_terms & normalized_qterms)
            if overlap >= 2:
                score += min(28.0, 12.0 + 5.0 * overlap)
                why.append('matching table column headers')
        if match_strength and len(text.strip()) < 130:
            score -= 13
            why.append("heading-only page")
        # No exact anchor: only allow very strong table-specific lexical evidence.
        # Ordinary questions should rely on dense/sparse retrieval, not random pages.
        if not (match_strength or ids_found or plan.page_number is not None
                or 'matching table column headers' in why):
            score = 0
        if score >= 14:
            scored.append((score, i, page, "; ".join(why), best_heading))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def _extract_matching_section(page: dict[str, Any], heading: str | None, question: str,
                              max_chars: int) -> str:
    text = page.get("text", "").strip()
    if not text:
        return ""
    if heading:
        headings = extract_headings(text)
        matches = [(pos, end) for title, pos, end in headings if title == heading]
        if matches:
            start, _ = matches[-1]
            next_starts = [pos for _, pos, _ in headings if pos > start]
            stop = min(next_starts) if next_starts else len(text)
            # Avoid a spurious empty section caused by nested Markdown headings.
            excerpt = text[start:stop].strip()
            if len(excerpt) >= 60 or _has_table(excerpt):
                text = excerpt
    if len(text) <= max_chars:
        return text
    # At natural line boundaries to avoid slicing a table row in half.
    lines, length = [], 0
    for line in text.splitlines():
        if lines and length + len(line) + 1 > max_chars:
            break
        lines.append(line)
        length += len(line) + 1
    return "\n".join(lines).strip() + "\n[Excerpt truncated; ask a narrower question for the remaining content.]"


def _adjacent_continuation(current: dict[str, Any], next_page: dict[str, Any],
                           question: str) -> bool:
    if _page_key(current) == _page_key(next_page):
        return False
    if current.get("doc_id", current.get("filename")) != next_page.get("doc_id", next_page.get("filename")):
        return False
    if current.get("page") is None or next_page.get("page") != current["page"] + 1:
        return False
    later = next_page.get("text", "")
    before = current.get("text", "")
    if not later.strip():
        return False
    # Page has a clearly different top-level heading => not a continuation.
    headings = extract_headings(later)
    if headings and headings[0][1] < 80 and _heading_match(headings[0][0], tokens(question)) == 0:
        # An unrelated heading is not enough alone to reject; compare tables.
        if not (_has_table(before) and _has_table(later)):
            return False
    before_terms = _content_terms(before)
    later_terms = _content_terms(later)
    question_terms = _content_terms(question)
    shared = len(before_terms & later_terms)
    request_shared = len(question_terms & later_terms)
    # Continuation markers or repeating table headers are domain-agnostic.
    continuation = bool(re.search(r"\b(continued|continuation|cont\.)\b", later[:600], re.I))
    prior_table = _has_table(before)
    next_table = _has_table(later)
    table_overlap = prior_table and next_table and shared >= 3
    query_overlap = request_shared >= 2 and (request_shared >= len(question_terms) * 0.5 if question_terms else False)
    return bool(continuation or (table_overlap and (query_overlap or shared >= 5)))


def choose_page_anchors(pages: list[dict[str, Any]], question: str,
                        max_anchors: int = 3) -> list[tuple[dict[str, Any], str, str | None]]:
    """Select strong source anchors discovered from the *current* document."""
    plan = plan_query(question)
    candidates = _page_candidates(pages, question, plan)
    result: list[tuple[dict[str, Any], str, str | None]] = []
    seen = set()

    def add(p, reason, heading):
        key = _page_key(p)
        if key not in seen and len(result) < max_anchors:
            result.append((p, reason, heading))
            seen.add(key)

    for _, _, p, reason, heading in candidates:
        add(p, reason, heading)
        if len(result) >= max_anchors:
            break
    # Include genuine same-document table continuation on next page if useful.
    if plan.wants_list and result and len(result) < max_anchors:
        for p, reason, heading in list(result[:2]):
            index = next((i for i, x in enumerate(pages) if _page_key(x) == _page_key(p)), -1)
            if index >= 0 and index + 1 < len(pages):
                nxt = pages[index + 1]
                if _adjacent_continuation(p, nxt, question):
                    add(nxt, "adjacent table/section continuation", None)
    # If the *only* high-confidence result is a divider page, include next page.
    if result and len(result) < max_anchors:
        for p, reason, _ in list(result):
            if len(p.get("text", "").strip()) >= 130:
                continue
            index = next((i for i, x in enumerate(pages) if _page_key(x) == _page_key(p)), -1)
            if index >= 0 and index + 1 < len(pages):
                nxt = pages[index + 1]
                if nxt.get("doc_id", nxt["filename"]) == p.get("doc_id", p["filename"]):
                    add(nxt, "content following matching heading", None)
    return result[:max_anchors]


def _requested_section_anchors(pages: list[dict[str, Any]], question: str,
                               max_anchors: int) -> list[tuple[dict[str, Any], str, str | None]]:
    """Find *all* explicitly named source headings, including two on one page.

    Uses headings read from the current documents rather than a domain-specific
    regex or an LLM. For repeated headings, prefer substantive table/content
    over a short divider. Distinct headings are independent evidence targets;
    a page is not a unique section identifier.

    Requires two or more complete heading mentions in the question. A prefix
    match alone is intentionally insufficient: 'Section 4' must not turn every
    'Section 4: ...' subsection into a different requested comparison target.
    """
    if max_anchors < 2:
        return []
    query_words = tokens(question)
    best: dict[tuple[str, tuple[str, ...]], tuple[float, int, dict[str, Any], str]] = {}
    for page in pages:
        body = page.get('text', '')
        if not body.strip():
            continue
        for title, _, _ in extract_headings(body):
            if _heading_match(title, query_words) < 0.999:
                continue
            heading_words = tuple(tokens(title))
            if not heading_words:
                continue
            # Locate where the exact heading was named to retain question order.
            offset = next((i for i in range(len(query_words) - len(heading_words) + 1)
                           if tuple(query_words[i:i + len(heading_words)]) == heading_words), None)
            if offset is None:
                continue
            section = _extract_matching_section(page, title, question, 30000)
            if not section.strip():
                continue
            # A 20-character chapter divider should lose to the real table on
            # an earlier page with the same heading.
            focus = _focus_terms(question) - set(heading_words)
            seen = _content_terms(section)
            # Prefer the named section whose *body* addresses the question,
            # not an unrelated table beneath the same short heading.
            overlap = len(focus & seen)
            quality = ((4.0 if _has_table(section) else 0.0)
                       + min(len(section) / 1200.0, 3.0)
                       + 2.5 * min(overlap, 5))
            if len(section) < 100:
                quality -= 4.0
            identity = (str(page.get('doc_id', page.get('filename', ''))), heading_words)
            previous = best.get(identity)
            if previous is None or quality > previous[0]:
                best[identity] = (quality, offset, page, title)
    if len(best) < 2:
        return []
    choices = sorted(best.values(), key=lambda row: (row[1], -row[0],
                    str(row[2].get('filename', '')), str(row[2].get('locator', ''))))
    return [(page, 'explicitly requested section: ' + title, title)
            for _, _, page, title in choices[:max_anchors]]



def _identifier_anchors(pages: list[dict[str, Any]], question: str, plan: QueryPlan,
                        max_anchors: int = 3) -> list[tuple[dict[str, Any], str, str | None]]:
    """Reserve source pages containing *exact* question identifiers.

    A query can ask for a record in a summary table AND details elsewhere in
    the document. Matching a generic heading (e.g. 'Detailed Syllabus') is
    not evidence that the target record's details were retrieved. Prefer pages
    with the requested identifier, prioritizing other named query facets too.
    Neither identifiers nor source headings are predefined for any document.
    """
    if not plan.identifiers or max_anchors < 1:
        return []
    q_words = tokens(question)
    q_terms = _content_terms(question)
    # Keep compound labels named by the user, e.g. 'Section-B', 'Phase-2',
    # 'cash-flow'. Exclude the record IDs themselves; those are tested below.
    compounds = [tuple(tokens(m.group())) for m in re.finditer(
        r"\b[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9]+\b", question)]
    compounds = list(dict.fromkeys(c for c in compounds
                                   if c and '-'.join(c).upper() not in plan.identifiers))
    scored: list[tuple[float, int, dict[str, Any], list[str]]] = []
    for page_no, page in enumerate(pages):
        body = page.get('text', '')
        found = [identifier for identifier in plan.identifiers if re.search(
            r'(?<![A-Za-z0-9])' + re.escape(identifier) + r'(?![A-Za-z0-9])', body, re.I)]
        if not found:
            continue
        body_words = tokens(body)
        searchable = ' ' + ' '.join(body_words) + ' '
        compound_hits = sum((' ' + ' '.join(parts) + ' ') in searchable
                            for parts in compounds)
        overlap = len(q_terms & _content_terms(body))
        heading_match = max((_heading_match(title, q_words)
                             for title, _, _ in extract_headings(body)), default=0.0)
        score = (25 * len(found) + 14 * heading_match + 9 * compound_hits
                 + min(overlap, 12) + min(len(body) / 1100.0, 3.0))
        if len(body.strip()) < 150:
            score -= 12
        scored.append((score, page_no, page, found))
    scored.sort(key=lambda r: (-r[0], r[1]))
    return [(page, 'exact identifier evidence: ' + ', '.join(found), None)
            for _, _, page, found in scored[:max_anchors]]


def _is_thin_identifier_free_anchor(anchor: tuple[dict[str, Any], str, str | None],
                                    identifiers: tuple[str, ...]) -> bool:
    """A short heading/divider is not sufficient for a named-record query."""
    page, _, title = anchor
    if not title:
        return False
    text = _extract_matching_section(page, title, '', 2500)
    if _has_table(text) or len(text.strip()) >= 180:
        return False
    return not any(re.search(r'(?<![A-Za-z0-9])' + re.escape(code) + r'(?![A-Za-z0-9])',
                             text, re.I) for code in identifiers)


# Cross-encoder output is a ranking logit, not a calibrated probability.
# Keep this conservative, configurable gate separate from structured-source
# anchors. The default applies to ms-marco-MiniLM-L6-v2; evaluate it for
# any alternate reranker on a labeled document/question benchmark.
def _score_gate(scored: list[tuple[int, float]], use_reranker: bool) -> float | None:
    if not use_reranker or not scored:
        return None
    floor = float(os.getenv('RAG_RERANK_MIN_SCORE', '-2.5'))
    max_gap = max(0.0, float(os.getenv('RAG_RERANK_MAX_GAP', '5.0')))
    return max(floor, scored[0][1] - max_gap)


def _focus_terms(question: str) -> set[str]:
    generic = set(('compare calculate difference between each year years fiscal '
                   'using identify provide cite cited respective relevant source '
                   'sources report document figures figure values value both '
                   'give results result versus vs list what how which where').split())
    return _content_terms(question) - generic


def _relevant_anchor(page: dict[str, Any], reason: str, heading: str | None,
                     question: str, identifiers: tuple[str, ...]) -> bool:
    # Strong named records and explicit page locators must survive even if the
    # page contains dense tables or a reranker dislikes their formatting.
    if 'exact identifier' in reason or 'explicit page number' in reason:
        return True
    body = _extract_matching_section(page, heading, question, 6000)
    if not body.strip() or len(body.strip()) < 55:
        return False
    wanted = _focus_terms(question)
    heading_words = set(tokens(heading or ''))
    content_terms = _content_terms(body) - heading_words
    overlap = wanted & content_terms
    # One-word section labels (e.g. "Consolidated") often occur in reports
    # dozens of times. Their body should independently support the question.
    if heading and len(tokens(heading)) <= 1 and len(wanted - heading_words) >= 2:
        return len(overlap - heading_words) >= 2
    if 'matching table column headers' in reason:
        return len(overlap) >= 2
    return bool(overlap) or bool(heading and len(tokens(heading)) >= 2)


def _normalized_hash(text: str) -> bytes:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).digest()



def _compared_headings(pages: list[dict[str, Any]], question: str) -> tuple[str, str] | None:
    """Find two *document-derived* labels joined by 'and', 'vs' or 'versus'.

    The labels come from actual headings or table-header cells, not a built-in
    financial or syllabus vocabulary. Multiple question conjunctions are fine;
    only one whose two sides are recognized labels qualifies.
    """
    names: set[str] = set()
    for page in pages:
        content = page.get('text', '')
        for heading, _, _ in extract_headings(content):
            heading_tokens = tokens(heading)
            if 1 <= len(heading_tokens) <= 3:
                names.add(heading.casefold())
            if heading_tokens and len(heading_tokens[0]) >= 4:
                names.add(heading_tokens[0])
        for line in content.splitlines():
            if not line.lstrip().startswith('|'):
                continue
            cells = [re.sub(r'<[^>]+>', ' ', c).strip(' *#\t').casefold()
                     for c in line.strip().strip('|').split('|')]
            for cell in cells:
                cell_words = tokens(cell)
                if cell and 1 <= len(cell_words) <= 3 and not re.search(r'\d', cell):
                    names.add(re.sub(r'\s+', ' ', cell))
                # Year-labeled headers such as 'Standalone 2025' still
                # contribute the *document-derived* category label.
                if cell_words and len(cell_words[0]) >= 4:
                    names.add(cell_words[0])
    if not names:
        return None
    # Restrict to exact phrases around a conjunction; single-word labels are
    # preferred over broad headings that could swallow the metric name.
    for match in re.finditer(r'\b(?:and|vs\.?|versus)\b', question, flags=re.I):
        left = question[:match.start()].strip()
        right = question[match.end():].strip()
        left_names = [n for n in names if re.search(r'\b' + re.escape(n) + r'\s*$', left, re.I)]
        right_names = [n for n in names if re.match(r'^\s*' + re.escape(n) + r'\b', right, re.I)]
        if left_names and right_names:
            a = max(left_names, key=lambda n: len(tokens(n)))
            b = max(right_names, key=lambda n: len(tokens(n)))
            if a != b:
                return a, b
    return None


def _table_answer_rows(text: str, question: str, labels: tuple[str, str]) -> list[str]:
    """Find numerical table rows addressing the metric, not just headings.

    'Standalone' and 'Consolidated' are examples of *query-derived* labels;
    this also handles arbitrary departments, products, or other categories.
    """
    exclude = set(tokens(' '.join(labels))) | {
        'year', 'years', 'figure', 'figures', 'statement', 'statements',
        'financial',
    }
    focus = {w for w in _focus_terms(question) - exclude
             if not re.fullmatch(r'(?:fy)?(?:19|20)\d{2}', w)}
    # Numeric years are compared in the table cells, not treated as the metric.
    lines = text.splitlines()
    matches: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or re.fullmatch(r'[| :\-]+', stripped):
            continue
        words = _content_terms(stripped)
        intersection = words & focus
        if not intersection:
            continue
        # Two numerical cells (often two compared years) make this an
        # answer-bearing table row, rather than a paragraph mentioning revenue.
        values = re.findall(r'(?<!\w)-?\d[\d,]*(?:\.\d+)?(?!\w)', stripped)
        if len(values) < 2:
            continue
        # Require two specific metric words when the question supplies them.
        useful = {w for w in focus if len(w) > 2 and not re.fullmatch(r'fy\d+', w)}
        need = 2 if len(useful) >= 2 else 1
        if len(intersection & useful) < need:
            continue
        matches.append(stripped)
    return matches


def _paired_table_anchors(pages: list[dict[str, Any]], question: str,
                          top_k: int) -> list[tuple[dict[str, Any], str, str | None]]:
    """Reserve the actual rows for both sides of a comparison.

    Prioritizes separate source statements when identifiable, otherwise a
    single joint comparison table. A table of contents or an unrelated section
    that only mentions the two labels is never considered sufficient.
    """
    if top_k < 1 or not re.search(r'\b(compare|difference|between|versus|vs|both)\b', question, re.I):
        return []
    pair = _compared_headings(pages, question)
    if not pair:
        return []
    terms = _focus_terms(question)
    candidates: dict[str, list[tuple[float, int, dict[str, Any], str | None]]] = {
        pair[0]: [], pair[1]: [], 'joint': []}
    query_has_statements = any(w in terms for w in ('statement', 'statements'))
    for idx, page in enumerate(pages):
        body = page.get('text', '')
        if not _has_table(body):
            continue
        rows = _table_answer_rows(body, question, pair)
        if not rows:
            continue
        headings = extract_headings(body)
        head_words = {title.casefold(): title for title, _, _ in headings}
        column_lines = [line.casefold() for line in body.splitlines()
                        if ('|' in line and any(label in line.casefold() for label in pair))]
        # Joint table *columns*, not accidental word mentions in prose. This
        # avoids classifying 'standalone selling price' in a consolidated note
        # as an additional standalone revenue table.
        joint_header = any(all(re.search(r'\b' + re.escape(label) + r'\b', line)
                                   for label in pair) for line in column_lines)
        if joint_header:
            quality = 20 + 4 * min(3, len(rows)) + min(len(body), 5000) / 5000
            candidates['joint'].append((quality, idx, page, None))
            continue
        for label in pair:
            # A single short word may appear in unrelated narrative. Prefer
            # explicitly titled source sections and a nearby 'statement' cue
            # when that is what the user requested.
            title = next((original for normalized, original in head_words.items()
                          if re.search(r'\b' + re.escape(label) + r'\b', normalized)), None)
            # Only label -> statement: the reverse order can mean an entirely
            # different statement followed by 'standalone selling price'.
            scope_pattern = (r'\b' + re.escape(label) +
                             r'\b[^\n.,;]{0,35}\bstatements?\b')
            label_near_statement = bool(re.search(scope_pattern, body, re.I))
            # On multi-page documents the first financial statement may be
            # titled merely 'Statement of Profit and Loss'; its category is
            # supplied on a preceding page. Carry forward a *nearby*, explicit
            # scope marker rather than hallucinating the category from numbers.
            inherited_scope = False
            if not title and not label_near_statement and query_has_statements:
                for previous in reversed(pages[max(0, idx - 4):idx]):
                    if previous.get('doc_id', previous.get('filename')) != page.get('doc_id', page.get('filename')):
                        break
                    if (isinstance(previous.get('page'), int) and
                        isinstance(page.get('page'), int) and
                        page['page'] - previous['page'] > 4):
                        break
                    prev_body = previous.get('text', '')
                    if re.search(scope_pattern, prev_body, re.I):
                        # Require the current page to be a relevant statement,
                        # not just an arbitrary table following the marker.
                        inherited_scope = any('statement' in t.lower() for t, _, _ in headings)
                        break
            if not title and not label_near_statement and not inherited_scope:
                continue
            quality = (15 if title else 7) + (8 if label_near_statement else 0)
            if inherited_scope:
                quality += 12
            if query_has_statements and label_near_statement:
                quality += 8
            quality += 2 * min(3, len(rows))
            candidates[label].append((quality, idx, page, title))
    # Prefer separate independently scoped evidence for both categories,
    # except when a complete side-by-side table supplies them more directly.
    if all(candidates[label] for label in pair) and top_k >= 2:
        chosen = [max(candidates[label], key=lambda x: (x[0], -x[1])) for label in pair]
        if _page_key(chosen[0][2]) != _page_key(chosen[1][2]):
            result = []
            for label, (_, _, page, title) in zip(pair, chosen):
                if title and not _table_answer_rows(
                        _extract_matching_section(page, title, question, 10000), question, pair):
                    title = None  # A footer heading must not cut off the data row.
                result.append((page, 'answer-bearing comparison table: ' + label, title))
            return result
    if candidates['joint']:
        _, _, page, title = max(candidates['joint'], key=lambda x: (x[0], -x[1]))
        return [(page, 'complete side-by-side comparison table: ' + ' / '.join(pair), title)]
    return []

def retrieve_evidence(engine: "DocumentRAG", question: str, top_k: int = 8,
                      candidate_pool: int = 90, context_budget: int = 14300) -> list[dict[str, Any]]:
    if engine.index is None or engine.embeddings is None:
        raise ValueError("Build the index first")
    if not question.strip() or top_k < 1:
        raise ValueError("Question must be nonempty and top_k >= 1")
    plan = plan_query(question)
    # If the user names one or more uploaded files, constrain retrieval to those
    # files. This is discovered from the upload list, never a fixed filename.
    named_docs = {f'doc-{i}' for i, (name, _) in enumerate(engine.documents)
                  if name.casefold() in question.casefold()} if engine.documents else set()
    active_pages = [p for p in engine.pages
                    if not named_docs or p.get('doc_id') in named_docs]
    # On a comparison of two named categories, prefer actual metric rows in
    # correctly scoped statements (or a side-by-side table) over generic
    # headings, appendices, and scattered narrative mentions.
    paired_anchors = _paired_table_anchors(active_pages, question, top_k)
    if plan.page_number is not None:
        paired_anchors = [a for a in paired_anchors
                          if a[0].get('page') == plan.page_number]
    # A comparison can require several independent *sections*, even when
    # those sections appear on the same physical page. Preserve one source
    # excerpt per explicitly named heading before global dense/BM25 ranking.
    multi_anchors = _requested_section_anchors(active_pages, question,
                                                max_anchors=min(top_k, 8))
    identifier_anchors = _identifier_anchors(active_pages, question, plan,
                                             max_anchors=min(top_k, 3))
    if plan.page_number is not None:
        multi_anchors = [(p, reason, title) for p, reason, title in multi_anchors
                         if p.get('page') == plan.page_number]
        identifier_anchors = [(p, reason, title) for p, reason, title in identifier_anchors
                              if p.get('page') == plan.page_number]
    page_anchors = choose_page_anchors(active_pages, question,
                                       max_anchors=min(top_k, 3))
    # Combine named sections with exact-record evidence. A generic divider
    # heading cannot crowd out the page containing the named record's details.
    # Retain different named sections on the same page (not deduped by page).
    anchors: list[tuple[dict[str, Any], str, str | None]] = []
    used_sections: set[tuple[tuple[str, str], str]] = set()
    used_pages: set[tuple[str, str]] = set()
    prioritized = (list(paired_anchors) if paired_anchors else
                   [a for a in multi_anchors if not (plan.identifiers and
                    _is_thin_identifier_free_anchor(a, plan.identifiers))] + identifier_anchors)
    # When two distinct, strong source pages already contain the named record,
    # a generic adjacent-page heading/table is not additional evidence.
    strong_pages = {_page_key(a[0]) for a in prioritized}
    fallback = ([] if paired_anchors or (len(multi_anchors) >= 2 and not plan.identifiers)
                or (plan.identifiers and len(strong_pages) >= 2) else
                [a for a in page_anchors if not (plan.identifiers and
                 _is_thin_identifier_free_anchor(a, plan.identifiers))])
    for a in prioritized + fallback:
        p, _, title = a
        if plan.page_number is not None and p.get('page') != plan.page_number:
            continue
        key = _page_key(p)
        if (key, title or '') in used_sections or (not title and key in used_pages):
            continue
        # If a page has already been selected as a named section, avoid a
        # second whole-page excerpt of the same underlying page.
        if not title and key in used_pages:
            continue
        if not _relevant_anchor(p, a[1], title, question, plan.identifiers):
            continue
        anchors.append(a)
        used_sections.add((key, title or ''))
        used_pages.add(key)
        if len(anchors) >= min(top_k, 5):
            break
    query_vector = engine.encoder.encode_query(
        [engine.query_prefix + question], convert_to_numpy=True, normalize_embeddings=True)
    q = np.asarray(query_vector, dtype=np.float32)
    n = len(engine.chunks)
    limit = min(max(candidate_pool, top_k * 5), n)
    _, ids = engine.index.search(q, limit)
    accepted = lambda i: not named_docs or engine.chunks[i].get('doc_id') in named_docs
    dense_ids = [int(i) for i in ids[0] if 0 <= i < n and accepted(int(i))]
    bm25 = engine._bm25(question)
    sparse_ids = sorted((i for i in bm25 if accepted(i)), key=lambda i: (-bm25[i], i))[:limit]
    rrf = defaultdict(float)
    for ranking, weight in ((dense_ids, 1.0), (sparse_ids, 1.2)):
        for rank, i in enumerate(ranking, 1):
            rrf[i] += weight / (60.0 + rank)
    for i, ch in enumerate(engine.chunks):
        if not accepted(i):
            continue
        if plan.identifiers and any(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", ch["text"], re.I)
                                    for value in plan.identifiers):
            rrf[i] += 0.06
    pool = sorted(rrf, key=lambda i: (-rrf[i], i))[:min(80, n)]
    if engine.use_reranker and pool:
        pairs = [(question, engine.chunks[i]["text"][:2100]) for i in pool]
        raw = np.asarray(engine.reranker.predict(pairs, batch_size=16, show_progress_bar=False), dtype=np.float32).reshape(-1)
        if len(raw) != len(pool):
            raise ValueError("Reranker returned unexpected score count")
        scored = [(i, float(raw[j])) for j, i in enumerate(pool)]
        scored.sort(key=lambda t: (-t[1], -rrf[t[0]], t[0]))
    else:
        scored = [(i, float(rrf[i])) for i in pool]

    min_score = _score_gate(scored, engine.use_reranker)
    selected: list[dict[str, Any]] = []
    occupied = set()
    hashes = set()
    size = 0
    for position, (page, reason, heading) in enumerate(anchors):
        # Reserve space for *every* requested target instead of allowing the
        # first long section to consume the entire evidence budget.
        remaining = max(1, len(anchors) - position)
        available = min(7000, (context_budget - size) // remaining)
        if available < 300:
            break
        snippet = _extract_matching_section(page, heading, question, available)
        if not snippet:
            continue
        if len(snippet) + size > context_budget:
            continue
        key = _page_key(page)
        page_chunk_ids = [i for i, ch in enumerate(engine.chunks) if _page_key(ch) == key]
        similarity = max((float(np.dot(engine.embeddings[i], q[0])) for i in page_chunk_ids), default=0.)
        selected.append({**{k: page.get(k) for k in ("doc_id", "filename", "page", "locator", "kind")},
                         "text": snippet, "score": similarity, "rerank_score": None, "rrf_score": None,
                         "evidence_type": "source-section", "reason": reason,
                         "source_heading": heading})
        occupied.add(key)
        hashes.add(_normalized_hash(snippet))
        size += len(snippet)

    # If we have an answer-bearing paired comparison table (or an individual
    # source for each side), do not append low-value global-rank results.
    # Reranker logits only compare relevance; they are not evidence coverage.
    if paired_anchors and len(selected) == len(paired_anchors) and all(
            _table_answer_rows(x['text'], question, _compared_headings(active_pages, question))
            for x in selected):
        return selected

    # Respect explicit page requests: don't mix in unrelated pages.
    if plan.page_number is not None and selected:
        return selected

    # A complete, heading-matched table is more reliable evidence for a
    # list/comparison than unrelated high-ranked prose. Do not add noise when
    # the selected source already preserves the rows and header untruncated.
    # This is about *structural evidence*, not a specific document schema.
    if (multi_anchors and len(selected) == len(anchors) and
        all(_has_table(s['text']) and '[Excerpt truncated' not in s['text']
            for s in selected)):
        # All named targets are covered by complete structured source excerpts;
        # further global-ranking hits would only introduce unrelated noise.
        return selected
    if (not multi_anchors and not plan.identifiers and plan.wants_list and selected and
        any((('matching document heading' in s['reason'] or
              'matching table column headers' in s['reason'])
             and _has_table(s['text']) and '[Excerpt truncated' not in s['text'])
            for s in selected)):
        return selected

    per_source: Counter[tuple[str, str]] = Counter()
    selected_tokens: list[set[str]] = [_content_terms(s["text"]) for s in selected]
    for i, rr_score in scored:
        if len(selected) >= top_k:
            break
        if min_score is not None and rr_score < min_score:
            # Stop instead of padding citations with low-quality evidence.
            # When *all* cross-encoder logits are very poor and no structural
            # anchor exists, caller can return an evidence-not-found response.
            break
        ch = engine.chunks[i]
        key = _page_key(ch)
        if key in occupied:
            continue
        text = ch["text"].strip()
        if min_score is not None and not (_focus_terms(question) & _content_terms(text)):
            # A model-only match with zero query-specific lexical overlap is
            # too weak to promote into a cited passage by default.
            continue
        if size + len(text) > context_budget:
            continue
        if per_source[key] >= 2:
            continue
        digest = _normalized_hash(text)
        if digest in hashes:
            continue
        terms = _content_terms(text)
        # Diversification: don't return a near-duplicate, but don't remove
        # distinct rows/pages just because they share a short heading.
        if selected_tokens and len(terms) > 6 and max(len(terms & earlier) / max(1, len(terms))
                                                     for earlier in selected_tokens) >= 0.93:
            continue
        vector_sim = float(np.dot(engine.embeddings[i], q[0]))
        selected.append({**ch, "score": vector_sim, "rerank_score": rr_score,
                         "rrf_score": float(rrf[i]),
                         "evidence_type": ch.get("chunk_type", "chunk"),
                         "reason": "hybrid retrieval"})
        selected_tokens.append(terms)
        hashes.add(digest)
        per_source[key] += 1
        size += len(text)
    return selected
