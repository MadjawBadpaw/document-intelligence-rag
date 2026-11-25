"""Offline, document-neutral regression tests: no Groq/HF downloads required.

Simulates the same API surface as the eventual backend. Real PDF/DOCX containers
are parsed; embedding, reranker and OCR model inference are mocked. Tests source
selection and structural integrity, NOT actual LLM answer faithfulness.

Run: python test_general_rag.py
Optional: python test_general_rag.py --syllabus "CSE (3).pdf"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


class FakeIndex:
    def __init__(self, dim):
        self.v = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors):
        self.v = np.asarray(vectors, dtype=np.float32)

    def search(self, queries, k):
        sims = self.v @ np.asarray(queries[0], dtype=np.float32)
        order = np.argsort(-sims, kind='stable')[:k]
        return sims[order].reshape(1, -1), order.reshape(1, -1)


def vector(text):
    text = text.removeprefix('Represent this sentence for searching relevant passages: ')
    v = np.zeros(384, dtype=np.float32)
    for word in re.findall(r'[a-z0-9]+', text.casefold()):
        v[int(hashlib.sha1(word.encode()).hexdigest()[:8], 16) % len(v)] += 1.0
    return v / max(1e-12, np.linalg.norm(v))


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode_document(self, texts, **kwargs):
        return np.asarray([vector(t) for t in texts], dtype=np.float32)

    def encode_query(self, texts, **kwargs):
        return np.asarray([vector(t) for t in texts], dtype=np.float32)


class MisleadingReranker:
    """Deliberately favors prose over tables to test anchor robustness."""
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, **kwargs):
        return np.asarray([(-35 if '|' in text or 'CREDIT' in text else 25)
                           + 0.05 * sum(w in text.lower() for w in re.findall(r'[a-z]+', q.lower()))
                           for q, text in pairs], dtype=np.float32)


sys.modules['faiss'] = types.SimpleNamespace(IndexFlatIP=FakeIndex)
sys.modules['sentence_transformers'] = types.SimpleNamespace(
    SentenceTransformer=FakeEncoder, CrossEncoder=MisleadingReranker)

import pymupdf
from docx import Document
from docx.shared import Inches
from PIL import Image, ImageDraw
from rag import DocumentRAG, _is_overview, _is_summary
from rag_chunking import chunk_pages
from rag_retrieval import plan_query, extract_headings


def make_pdf(pages: list[str]) -> bytes:
    pdf = pymupdf.open()
    try:
        for content in pages:
            page = pdf.new_page(width=612, height=792)
            for j, line in enumerate(content.splitlines()[:42]):
                page.insert_text((30, 45 + j * 16), line[:110], fontsize=10)
        return pdf.tobytes()
    finally:
        pdf.close()


def assert_citations_and_table_sources():
    report_pages = [
        '# Annual Financial Report\nExecutive summary of operations and performance',
        'FY2025 Regional Results\n| Region | Revenue ($m) | Margin (%) |\n'
        '| --- | --- | --- |\n| North | 170 | 24 |\n| South | 90 | 16 |\n| West | 40 | 8 |',
        'Regional Results Continued\n| Region | Revenue ($m) | Margin (%) |\n'
        '| --- | --- | --- |\n| East | 60 | 12 |\n| Overseas | 25 | 7 |',
        'Risk Factors\nCybersecurity risk and vendor concentration.',
        'Appendix B - Audit Evidence\nINV-2025-009 was checked and signed.',
    ]
    pages = [{'filename':'finance.md','doc_id':'finance','page':i+1, 'locator':f'page {i+1}',
              'kind':'pdf','text':t,'has_ocr':False} for i,t in enumerate(report_pages)]
    engine = DocumentRAG(ocr='off', rerank=True)
    engine.pages = pages
    engine.chunks = chunk_pages(pages, 1000, 120)
    engine.embeddings = FakeEncoder().encode_document([p['text'] for p in engine.chunks])
    engine._setup_index()
    out = engine.retrieve('List all values in FY2025 Regional Results with revenues and margins', 8)
    assert out and out[0]['page'] == 2, [(x['page'],x['reason']) for x in out]
    assert all(word in out[0]['text'] for word in ('North', 'South', 'West', '170', '24'))
    assert any(x['page'] == 3 for x in out), [(x['page'],x['reason']) for x in out]
    code = engine.retrieve('Where is INV-2025-009 discussed?', 4)
    assert code[0]['page'] == 5 and 'INV-2025-009' in code[0]['text']
    assert not _is_summary('List all values in FY2025 Regional Results')
    assert _is_overview('overview of the document')
    assert _is_summary('Summarize the whole document')
    assert not _is_summary('Summarize Chapter 4')
    print('[PASS] Finance: multi-page table, adversarial reranker, source locator, IDs')


def assert_multi_section_coverage():
    """Two targets can occupy one PDF page: never de-duplicate by page alone."""
    pages = [
        {'filename': 'inventory.pdf', 'doc_id': 'doc-0', 'page': 1, 'locator': 'page 1',
         'kind': 'pdf', 'has_ocr': False,
         'text': '# Branch Alpha\n| Item | Stock |\n| --- | --- |\n'
                 '| Bolts | 160 |\n| Nails | 250 |\n'
                 '# Branch Beta\n| Item | Stock |\n| --- | --- |\n'
                 '| Bolts | 90 |\n| Nails | 300 |'},
        {'filename': 'inventory.pdf', 'doc_id': 'doc-0', 'page': 2, 'locator': 'page 2',
         'kind': 'pdf', 'has_ocr': False,
         'text': '# Branch Gamma\n| Item | Stock |\n| --- | --- |\n'
                 '| Bolts | 12 |\n| Nails | 34 |'},
        {'filename': 'inventory.pdf', 'doc_id': 'doc-0', 'page': 3, 'locator': 'page 3',
         'kind': 'pdf', 'has_ocr': False, 'text': '# Miscellaneous\nOther notes.'},
    ]
    engine = DocumentRAG(ocr='off', rerank=True, device='cpu')
    engine.pages = pages
    engine.chunks = chunk_pages(pages, 1000, 120)
    engine.embeddings = FakeEncoder().encode_document([x['text'] for x in engine.chunks])
    engine._setup_index()

    q = 'Compare Branch Alpha and Branch Beta item stocks and totals.'
    hits = engine.retrieve(q, top_k=8)
    names = [h.get('source_heading') for h in hits]
    assert names == ['Branch Alpha', 'Branch Beta'], [(h['page'], h['reason']) for h in hits]
    assert [h['page'] for h in hits] == [1, 1], hits
    assert '160' in hits[0]['text'] and '250' in hits[0]['text']
    assert '90' in hits[1]['text'] and '300' in hits[1]['text']
    assert '90' not in hits[0]['text'] and '160' not in hits[1]['text']

    q = 'List Branch Alpha, Branch Beta and Branch Gamma item stocks.'
    hits = engine.retrieve(q, top_k=8)
    assert [h.get('source_heading') for h in hits] == [
        'Branch Alpha', 'Branch Beta', 'Branch Gamma'], hits
    assert [h['page'] for h in hits] == [1, 1, 2], hits
    assert all('|' in h['text'] for h in hits)

    # Markdown emitted by modern PDF layout extraction can encode headings
    # with bold markup; coverage must still preserve both table sections.
    md = [
        {'filename': 'layout.pdf', 'doc_id': 'doc-1', 'page': 8, 'locator': 'page 8',
         'kind': 'pdf', 'has_ocr': False,
         'text': '###### **Chapter 2**\n| Part | Cost |\n| --- | --- |\n'
                 '| Engine | 115 |\n| Pump | 16 |\n'
                 '###### **Chapter 5**\n| Part | Cost |\n| --- | --- |\n'
                 '| Engine | 215 |\n| Pump | 26 |'}
    ]
    layout = DocumentRAG(ocr='off', rerank=True, device='cpu')
    layout.pages = md
    layout.chunks = chunk_pages(md, 1000, 120)
    layout.embeddings = FakeEncoder().encode_document([x['text'] for x in layout.chunks])
    layout._setup_index()
    hits = layout.retrieve('Compare Chapter 2 and Chapter 5 costs', top_k=8)
    assert [h.get('source_heading') for h in hits] == ['Chapter 2', 'Chapter 5'], hits
    assert '115' in hits[0]['text'] and '215' in hits[1]['text']

    # Headings are document-derived, not hardcoded for syllabus/semester names.
    q = 'What is the stock in Branch Beta?'
    hits = engine.retrieve(q, top_k=8)
    assert hits and '90' in hits[0]['text'] and '300' in hits[0]['text']
    print('[PASS] Multi-section: independent evidence for two on one page, three across pages, adversarial reranker')



def assert_cross_page_record_details():
    """Named record requires metadata and detailed facets on distant pages.

    A generic matching divider must not displace the detailed record, even
    with an adversarial reranker. No syllabus-specific retrieval rules.
    """
    records = [
        {'filename':'plant_manual.pdf', 'doc_id':'doc-0', 'page':1, 'locator':'page 1',
         'kind':'pdf','has_ocr':False,
         'text':'# Product Matrix\n| Code | Product | Hours |\n| --- | --- | --- |\n'
                '| DEV-420 | Laser Controller | 8 |'},
        {'filename':'plant_manual.pdf', 'doc_id':'doc-0', 'page':2, 'locator':'page 2',
         'kind':'pdf','has_ocr':False,
         'text':'# Detailed Specifications\nTechnical department\nPage 2'},
        {'filename':'plant_manual.pdf', 'doc_id':'doc-0', 'page':3, 'locator':'page 3',
         'kind':'pdf','has_ocr':False,
         'text':'# Unrelated Components\nInventory items without the target.'},
        {'filename':'plant_manual.pdf', 'doc_id':'doc-0', 'page':95, 'locator':'page 95',
         'kind':'pdf','has_ocr':False,
         'text':'Device specifications for the Laser Controller DEV-420.\n'
                '| Phase | Tasks |\n| --- | --- |\n'
                '| Stage-A | Input calibration, threshold configuration |\n'
                '| Stage-B | Sensor reconciliation, final validation |'},
    ]
    e = DocumentRAG(ocr='off', rerank=True, device='cpu')
    e.pages = records
    e.chunks = chunk_pages(records, 1000, 120)
    e.embeddings = FakeEncoder().encode_document([c['text'] for c in e.chunks])
    e._setup_index()
    q = ('Using the Product Matrix and Detailed Specifications for DEV-420, '
         'identify the product hours and list tasks in Stage-A and Stage-B. Cite pages.')
    hits = e.retrieve(q, top_k=8)
    assert [h['page'] for h in hits[:2]] == [1, 95], [
        (h['page'], h['reason']) for h in hits]
    assert all(s in hits[1]['text'] for s in ('Stage-A', 'Stage-B',
                                              'Input calibration', 'final validation'))
    assert not any(h['page'] == 2 for h in hits[:2]), hits
    print('[PASS] Cross-page record: exact ID detail beats generic divider and adversarial reranker')


def assert_native_pdf_and_long_document(tmp: Path):
    text = [
        'Section 1\nIntroduction and mission statement',
        'Section 2\nOperations and users',
        'Section 3\nAPI configuration and limits',
        'Section 4\nDeployment Procedure\nStep 1: Validate config\nStep 2: Deploy\nStep 3: Verify',
        'Section 5\nTroubleshooting',
    ]
    pdf = make_pdf(text)
    e = DocumentRAG(cache_dir=tmp / 'manual-cache', ocr='off', rerank=True)
    e.build_index([('manual.pdf', pdf)])
    result = e.retrieve('Explain Section 4 deployment steps', 8)
    assert result and any(x['page'] == 4 and 'Validate config' in x['text'] for x in result)
    assert e.stats()['pdf_pages'] == 5
    # Cache re-use (same fingerprint + without refreshing vectors)
    f = DocumentRAG(cache_dir=tmp / 'manual-cache', ocr='off', rerank=False)
    f.build_index([('manual.pdf', pdf)])
    assert f.stats()['pdf_pages'] == 5
    assert f.retrieve('Step 3 verify deployment', 3)
    print('[PASS] Manual: ordinary text PDF, section retrieval, persistent cache')

    large = []
    for i in range(1, 322):
        prefix = f'Section {i}\nEngineering notes on system {i}.'
        if i == 314:
            prefix += '\nLate-page recovery includes the token ORCHID-3140.'
        large.append(prefix)
    bigpdf = make_pdf(large)
    g = DocumentRAG(cache_dir=tmp / 'large-cache', ocr='off', rerank=False)
    g.build_index([('large_manual.pdf', bigpdf)])
    assert g.stats()['pdf_pages'] == 321
    found = g.retrieve('Find ORCHID-3140 on the late page', 5)
    assert found and found[0]['page'] == 314, [(s['page'],s['text'][:50]) for s in found]
    print('[PASS] Long PDF: 321 pages, indexed and late-page retrieval')


def assert_table_split_and_docx(tmp: Path):
    table = '| Item | Manufacturer | Cost |\n|---|---|---|\n' + '\n'.join(
        f'| ITEM-{i:04d} | Firm {i} | {i * 11} |' for i in range(185))
    chunks = chunk_pages([{'filename':'prices.md', 'doc_id':'prices','locator':'text',
                           'page':None,'kind':'text','text':'# Procurement Ledger\n'+table}], 1000, 120)
    fragments = [c for c in chunks if c['chunk_type']=='table']
    assert len(fragments) > 1
    assert all('| Item | Manufacturer | Cost |' in c['text'] for c in fragments)
    assert all(f'ITEM-{i:04d}' in '\n'.join(c['text'] for c in fragments) for i in range(185))
    print('[PASS] Large table: 185 rows retained, repeated header, no row lost')

    doc = Document()
    doc.add_heading('Warranty Agreement', level=1)
    for i in range(17):
        doc.add_paragraph(f'General contract paragraph {i}: routine terms.')
    doc.add_heading('Article 9 Termination', level=2)
    doc.add_paragraph('The warranty lasts 24 months after delivery.')
    cells = doc.add_table(rows=3, cols=2)
    cells.cell(0,0).text='Tier'; cells.cell(0,1).text='Coverage'
    cells.cell(1,0).text='Basic'; cells.cell(1,1).text='24 months'
    cells.cell(2,0).text='Premium'; cells.cell(2,1).text='36 months'
    b=io.BytesIO(); doc.save(b)
    backend=DocumentRAG(cache_dir=tmp/'docx-cache',ocr='off',rerank=True)
    backend.build_index([('agreement.docx',b.getvalue())])
    hits=backend.retrieve('What does Article 9 say about termination and warranty duration?',8)
    assert any('24 months' in x['text'] for x in hits)
    assert any('agreement.docx' == x['filename'] and 'block' in x['locator'] for x in hits)
    assert any('Premium' in p['text'] for p in backend.pages)
    print('[PASS] DOCX: heading, table, grouped paragraphs, block citations')


def assert_ocr_wiring():
    pic=Image.new('RGB',(900,200),'white')
    ImageDraw.Draw(pic).text((25,70),'ALERT FROM SCANNED IMAGE',fill='black')
    b=io.BytesIO();pic.save(b,format='PNG'); png=b.getvalue()
    doc=pymupdf.open()
    try:
        a=doc.new_page(width=650,height=260)
        a.insert_image(pymupdf.Rect(0,0,640,250),stream=png)
        pdf=doc.tobytes()
    finally:
        doc.close()
    ocr=DocumentRAG(ocr='auto',rerank=False)
    ocr._ocr_image=lambda content, label: 'ALERT FROM SCANNED IMAGE'
    extracted=ocr.extract_pdf(pdf,'raster.pdf')
    assert extracted[0]['has_ocr'] and 'ALERT FROM SCANNED IMAGE' in extracted[0]['text']
    word=Document();word.add_paragraph('See attached:').add_run().add_picture(io.BytesIO(png),width=Inches(2))
    out=io.BytesIO();word.save(out)
    records=ocr.extract_docx(out.getvalue(),'images.docx')
    assert any('ALERT FROM SCANNED IMAGE' in p['text'] for p in records)
    print('[PASS] Scanned PDF and DOCX images (mock OCR inference, real containers)')


def assert_relevance_gate_and_competing_tables():
    """No arbitrary padding; named parallel statements do not cross-contaminate."""
    rows = [
        '# Consolidated\nGeneral workforce survey and hiring details.\n'
        '| Staff | Response |\n| --- | --- |\n| Employees | 7000 |',
        '# Standalone\nRevenue from operations and FY2025 / FY2024 breakdown.\n'
        '| Category | FY2025 | FY2024 |\n| --- | --- | --- |\n'
        '| Revenue from operations | 100 | 90 |',
        '# Consolidated\nRevenue from operations and FY2025 / FY2024 breakdown.\n'
        '| Category | FY2025 | FY2024 |\n| --- | --- | --- |\n'
        '| Revenue from operations | 140 | 125 |',
        '# Unrelated Appendix\nA historical summary of capital equipment and banking.',
    ]
    pages = [dict(filename='finance.pdf', doc_id='doc-0', page=i+1,
                  locator=f'page {i+1}', kind='pdf', text=body, has_ocr=False)
             for i, body in enumerate(rows)]
    e=DocumentRAG(ocr='off',rerank=True, device='cpu')
    e.pages=pages
    e.chunks=chunk_pages(pages, 1000, 120)
    e.embeddings=FakeEncoder().encode_document([ch['text'] for ch in e.chunks])
    e._setup_index()
    q=('Compare the Standalone and Consolidated revenue from operations '
       'in FY2025 and FY2024 and cite both statements.')
    hits=e.retrieve(q, 8)
    assert [(h.get('source_heading'), h['page']) for h in hits] == [
        ('Standalone', 2), ('Consolidated', 3)], [(h.get('source_heading'),h['page']) for h in hits]
    assert '100' in hits[0]['text'] and '140' in hits[1]['text']

    # A named-record search should prefer its answer even when most corpus
    # passages have superficial lexical overlap and low reranker scores.
    class ScoreReranker:
        def predict(self, pairs, **kwargs):
            return np.asarray([4.0 if '24 months' in body else -8.5
                               for _,body in pairs], dtype=np.float32)
    e._reranker = ScoreReranker()
    e.pages = [dict(filename='manual.pdf',doc_id='doc-0',page=i+1,
                    locator=f'page {i+1}',kind='pdf',text=t,has_ocr=False)
               for i,t in enumerate([
                 'Service policies\nThe warranty lasts 24 months from delivery.',
                 'Business process review\nPricing, legal and logistics.',
                 'Marketing overview\nThe warranty program campaign had a successful launch.',
                 'General questions\nTroubleshooting and maintenance.',
               ])]
    e.chunks = chunk_pages(e.pages,1000,120)
    e.embeddings = FakeEncoder().encode_document([ch['text'] for ch in e.chunks])
    e._setup_index()
    q='What is the warranty duration in the manual?'
    hits=e.retrieve(q,8)
    assert len(hits)==1 and '24 months' in hits[0]['text'], hits

    # When every passage scores very poorly, return no evidence instead of
    # attaching irrelevant citations to a refusal.
    e._reranker = type('PoorReranker', (), {'predict':lambda self,pairs,**k:
                               np.full(len(pairs), -8.0, dtype=np.float32)})()
    hits=e.retrieve('What was the 2025 Google placement package?',8)
    assert not hits, hits
    e._generate=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Should not contact LLM'))
    refused=e.ask('What was the 2025 Google placement package?')
    assert not refused['sources'] and 'find relevant text' in refused['answer']
    print('[PASS] Relevance: irrelevant citations removed; paired financial tables isolated')



def assert_report_comparison_coverage():
    """A real annual report failure pattern: numbers on distant statement pages.

    Synthetic page content mirrors structure, NOT a live annual-report benchmark.
    Generic section names and narrative passages must not crowd out the rows.
    """
    cases = [
        (8, '# Financial statements\nStandalone 196. Consolidated 276.\n'
            'Index of statements and explanatory notes.'),
        (30, '# Board report\n| Particulars | Standalone 2025 | Standalone 2024 |'
             ' Consolidated 2025 | Consolidated 2024 |\n| --- | --- | --- | --- | --- |\n'
             '| Revenue from operations | 136592 | 128933 | 162990 | 153670 |'),
        (34, '# Board report\n## Standalone\nCapital expenditure on equipment and services\n'
             '| Item | Value |\n| --- | --- |\n| Vehicles | 24 |'),
        (73, 'Revenue and performance overview. Selling and marketing expenses '
             'as a percentage of revenue on standalone and consolidated basis rose.'),
        (195, '# Balance Sheet\nThe accompanying notes form part of the Standalone financial statements.'),
        (196, '# Statement of Profit and Loss\n| Particulars | FY2025 | FY2024 |\n'
              '| --- | --- | --- |\n| Revenue from operations | 136592 | 128933 |\n'
              'Revenue is shown net of indirect taxes.'),
        (276, '# Consolidated Statement of Profit and Loss\n'
              '| Particulars | FY2025 | FY2024 |\n| --- | --- | --- |\n'
              '| Revenue from operations | 162990 | 153670 |'),
        (321, '# Revenue from operations\nThe Group presents revenues in its '
              'Consolidated Statement of Profit and Loss. Standalone selling price '
              'is evaluated for individual contracts.\n'
              '| Particulars | 2025 | 2024 |\n| --- | --- | --- |\n'
              '| Total revenue from operations | 162990 | 153670 |'),
    ]
    pages = [dict(filename='report.pdf', doc_id='doc-0', page=page,
                  locator=f'page {page}', kind='pdf', has_ocr=False, text=body)
             for page,body in cases]
    e=DocumentRAG(ocr='off', rerank=True, device='cpu')
    e.pages=pages
    e.chunks=chunk_pages(pages, 1000, 120)
    e.embeddings=FakeEncoder().encode_document([ch['text'] for ch in e.chunks])
    e._setup_index()
    class WrongReranker:
        def predict(self, pairs, **kwargs):
            return np.asarray([10.0 if 'selling and marketing expenses' in body.lower()
                               else (8.0 if 'Capital expenditure' in body
                                     else (-4.0 if '|' in body else 1.0))
                               for _,body in pairs],dtype=np.float32)
    e._reranker=WrongReranker()
    query=('Compare Infosys standalone and consolidated revenue from operations '
           'for FY2025 and FY2024, calculate their differences and cite the '
           'respective financial statements.')
    hits=e.retrieve(query, top_k=8)
    report=[(h['page'],h['reason']) for h in hits]
    assert len(hits) == 2, report
    assert [h['page'] for h in hits] == [196, 276], report
    assert all('Revenue from operations' in h['text'] for h in hits), report
    assert '136592' in hits[0]['text'] and '128933' in hits[0]['text']
    assert '162990' in hits[1]['text'] and '153670' in hits[1]['text']
    assert all(h['page'] not in (8, 34, 73, 321) for h in hits)

    # No independently scoped statements: a joint table must satisfy both
    # requested categories without unrelated supplemental sources.
    e.pages=[p for p in pages if p['page'] not in (196,276)]
    e.chunks=chunk_pages(e.pages,1000,120)
    e.embeddings=FakeEncoder().encode_document([ch['text'] for ch in e.chunks])
    e._setup_index()
    hits=e.retrieve(query,top_k=8)
    assert len(hits)==1 and hits[0]['page']==30, [
        (h['page'], h['reason']) for h in hits]
    assert all(str(v) in hits[0]['text'] for v in (136592,128933,162990,153670))
    print('[PASS] Annual-report-shaped comparison: source scope, numeric rows, no padding')

def assert_selective_inline_ocr():
    import random
    rng=random.Random(314159)
    photo=Image.new('RGB',(360,220))
    photo.putdata([(rng.randrange(256),rng.randrange(256),rng.randrange(256))
                   for _ in range(360*220)])
    diagram=Image.new('RGB',(360,220),'white')
    ImageDraw.Draw(diagram).text((15,35),'IMAGE TABLE: 2025=120 2024=100',fill='black')
    def png_bytes(img):
        b=io.BytesIO();img.save(b,format='PNG');return b.getvalue()
    pic=png_bytes(photo); sketch=png_bytes(diagram)
    pdf=pymupdf.open()
    try:
        for _ in range(2):
            pg=pdf.new_page(width=612,height=792)
            pg.insert_text((30,55),'Native selectable text ' * 10,fontsize=11)
            pg.insert_image(pymupdf.Rect(30,115,330,290),stream=pic)
            pg.insert_image(pymupdf.Rect(30,320,330,495),stream=sketch)
        pg=pdf.new_page(width=612,height=792)
        pg.insert_image(pymupdf.Rect(30,30,570,750),stream=sketch)
        blob=pdf.tobytes()
    finally:
        pdf.close()
    e=DocumentRAG(ocr='auto',rerank=False)
    visited=[]
    def ocr(img, label):
        visited.append(label)
        return 'RECOVERED IMAGE TABLE' if 'image' in label else 'SCANNED CONTENT'
    e._ocr_image=ocr
    pages=e.extract_pdf(blob,'mixed.pdf')
    assert len(pages)==3
    assert len(visited)==2,visited  # one repeated diagram, one full scanned page
    assert 'RECOVERED IMAGE TABLE' in pages[0]['text']
    assert pages[1]['has_ocr'] and 'RECOVERED IMAGE TABLE' in pages[1]['text']
    assert 'SCANNED CONTENT' in pages[2]['text']
    print('[PASS] Selective OCR: photos skipped, repeated diagrams deduped, scans kept')


def assert_modern_layout_route():
    """Exercise PyMuPDF 2.x style Document.to_markdown without installing it."""
    pdf_bytes = make_pdf(['Heading Alpha\nValue 11', 'Heading Beta\nValue 22',
                          'Heading Gamma\nValue 33'])
    native_open = pymupdf.open
    calls = []

    class LayoutProxy:
        def __init__(self, real):
            self.real = real
            self.needs_pass = real.needs_pass
        def __enter__(self): return self
        def __exit__(self, *args): self.real.close()
        def __len__(self): return len(self.real)
        def __iter__(self): return iter(self.real)
        def __getitem__(self, i): return self.real[i]
        def to_markdown(self, **kwargs):
            calls.append(kwargs['pages'])
            return [{'text': '## Modern parser page ' + str(i + 1) + '\n' +
                     self.real[i].get_text('text')}
                    for i in kwargs['pages']]

    try:
        pymupdf.open = lambda *args, **kwargs: LayoutProxy(native_open(*args, **kwargs))
        e = DocumentRAG(ocr='off', rerank=False)
        result = e.extract_pdf(pdf_bytes, 'modern.pdf')
    finally:
        pymupdf.open = native_open
    assert calls == [[0, 1, 2]], calls
    assert len(result) == 3 and result[2]['text'].startswith('## Modern parser page 3')
    print('[PASS] PyMuPDF 2.x Document.to_markdown route (mocked API, real PDF pages)')


def assert_generic_summary_and_client(tmp: Path):
    source = make_pdf(['Executive Overview\nPerformance improved.',
                       'Section 2\nOperational reliability increased.',
                       'Section 3\nCosts reduced.'])
    backend = DocumentRAG(cache_dir=tmp/'overview-cache', ocr='off', rerank=False)
    backend.build_index([('generic.pdf', source)])
    logs = []
    def fake_generate(system, user, model, max_tokens=1500, effort='low'):
        logs.append((system, user, model))
        return 'Grounded summary of provided evidence.'
    backend._generate = fake_generate
    first = backend.overview_document()
    second = backend.overview_document()
    assert first == second and len(logs) == 1
    detail = backend.summarize_document()
    assert 'Coverage' in detail and any('Operational reliability' in u for _, u, _ in logs)
    assert len(logs) >= 2
    answer = backend.ask('What happened to operational reliability?')
    assert answer['sources'] and answer['mode'] == 'retrieval'
    assert backend.inspect('operational reliability')
    print('[PASS] Generic overview cache, complete small-document summary, ask, inspect')


def assert_syllabus_if_given(path: Path,tmp:Path):
    if not path.exists():
        print('[SKIP] Optional external real-world PDF test: file not provided')
        return
    e=DocumentRAG(cache_dir=tmp/'syllabus-cache',ocr='off',rerank=True)
    e.build_index([(path.name,path.read_bytes())])
    assert e.stats()['pdf_pages']==160, e.stats()
    question='List all Semester VIII subjects with their course codes and credits.'
    hits=e.retrieve(question,top_k=8)
    print('real PDF hits:',[(h['page'],h['evidence_type'],h['reason']) for h in hits])
    assert hits[0]['page']==6, [(h['page'],h['reason']) for h in hits]
    codes=('CS-8001','CS-OE','CS-PE-II','CS-8051','CS-8052','HS-8001')
    assert all(code in hits[0]['text'] for code in codes),hits[0]['text']
    assert '17' in hits[0]['text']
    comparison = e.retrieve('Compare Semester VII and Semester VIII. List their subjects, '
                            'course codes, credits, and total credits for each semester.', top_k=8)
    assert [x.get('source_heading') for x in comparison] == ['Semester-VII', 'Semester-VIII'], [
        (x['page'], x.get('source_heading'), x['reason']) for x in comparison]
    assert comparison[0]['page'] == comparison[1]['page'] == 6
    assert all(c in comparison[0]['text'] for c in ('CS-7001', 'CS-7002', 'CS-7003', 'CS-7051', 'CS-7052', 'CS-7053'))
    assert all(c in comparison[1]['text'] for c in codes)
    assert 'CS-8001' not in comparison[0]['text'] and 'CS-7001' not in comparison[1]['text']
    # A multi-part question names both a summary heading and a generic
    # 'Detailed Syllabus' divider; source record details are 130 pages later.
    cross = e.retrieve('Using the Semester VIII course-structure table and the '
                       'detailed syllabus for CS-8001, identify its credits and '
                       'list all topics in Section-A and Section-B.', top_k=8)
    assert any(h['page'] == 6 and 'CS-8001' in h['text'] for h in cross), [
        (h['page'], h['reason']) for h in cross]
    assert any(h['page'] == 136 and 'Section-A' in h['text'] and
               'Section-B' in h['text'] and 'apriori' in h['text'].lower()
               for h in cross), [(h['page'], h['reason']) for h in cross]
    assert not any(h['page'] == 8 for h in cross[:2]), [
        (h['page'], h['reason']) for h in cross]
    print('[PASS] Optional real syllabus: same-page comparison AND cross-page detailed record')



def assert_device_labels():
    """Torch device labels should not break first indexing on Windows."""
    from unittest.mock import patch
    from rag import _resolve_device
    import torch
    assert _resolve_device(None) is None
    assert _resolve_device('auto') is None
    assert _resolve_device('CPU') == 'cpu'
    with patch.object(torch.cuda, 'is_available', return_value=True), \
         patch.object(torch.cuda, 'device_count', return_value=1):
        assert _resolve_device('gpu') == 'cuda'
        assert _resolve_device('CUDA') == 'cuda'
        assert _resolve_device('gpu:0') == 'cuda:0'
        assert _resolve_device('cuda:9') == 'cpu'
    with patch.object(torch.cuda, 'is_available', return_value=False):
        assert _resolve_device('gpu') == 'cpu'
        assert _resolve_device('cuda') == 'cpu'
    try:
        _resolve_device('graphics-card')
    except ValueError as err:
        assert 'Invalid RAG device' in str(err)
    else:
        raise AssertionError('Unknown device should give a clear error')
    print('[PASS] Device selection: gpu alias, CUDA availability, CPU fallback, invalid input')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--syllabus',type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as name:
        tmp=Path(name)
        assert_device_labels()
        assert_citations_and_table_sources()
        assert_multi_section_coverage()
        assert_cross_page_record_details()
        assert_native_pdf_and_long_document(tmp)
        assert_table_split_and_docx(tmp)
        assert_ocr_wiring()
        assert_selective_inline_ocr()
        assert_relevance_gate_and_competing_tables()
        assert_report_comparison_coverage()
        assert_modern_layout_route()
        assert_generic_summary_and_client(tmp)
        if args.syllabus:
            assert_syllabus_if_given(args.syllabus,tmp)
    print('ALL OFFLINE TESTS PASSED (live Groq, real ML and real OCR inference not exercised).')


if __name__=='__main__':
    main()
