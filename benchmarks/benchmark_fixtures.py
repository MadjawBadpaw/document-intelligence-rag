"""Generate deterministic synthetic RAG evaluation fixtures (not real records)."""
from __future__ import annotations

import argparse
from pathlib import Path


def generate(outdir: str | Path) -> list[Path]:
    from docx import Document
    from docx.shared import Inches, Pt
    from PIL import Image, ImageDraw, ImageFont

    dest = Path(outdir)
    dest.mkdir(parents=True, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    doc.add_heading('Security Operations Incident Runbook', level=0)
    doc.add_paragraph('Synthetic benchmark fixture. Revision RB-2025-06; effective 12 June 2025.')
    doc.add_heading('Alert severity and response', level=1)
    doc.add_paragraph('Encoded PowerShell: classify suspicious PowerShell executions containing encoded commands as High severity until investigated.')
    tb=doc.add_table(rows=1,cols=3)
    tb.style='Table Grid'
    for c,t in zip(tb.rows[0].cells, ['Severity','Initial triage SLA','Escalation target']): c.text=t
    for vals in [['Critical','5 minutes','Incident Commander'],['High','15 minutes','SOC lead'],['Medium','2 hours','Queue owner'],['Low','1 business day','Queue owner']]:
        row=tb.add_row()
        for c,t in zip(row.cells,vals): c.text=t
    doc.add_heading('Evidence preservation', level=1)
    doc.add_paragraph('Before endpoint isolation, preserve volatile memory evidence (memory image), record active network connections and record the current UTC time. Isolation may proceed immediately if there is ongoing destructive activity.')
    doc.add_heading('Escalation', level=1)
    doc.add_paragraph('A confirmed credential compromise must be escalated to the Identity Response on-call team and the SOC lead; document ticket ID and evidence hashes.')
    doc.add_heading('Closure', level=1)
    doc.add_paragraph('An incident may be closed when containment, remediation, and recovery are recorded and the SOC lead approves the investigation summary.')
    docx=dest/'soc_runbook.docx';doc.save(docx)

    md=dest/'api_reference.md'
    md.write_text('''# Events API Reference (Synthetic Fixture)\n\n## GET /v1/events\n\nRetrieve security events. Default page size: **50** events; maximum: **200**.\nA `cursor` value is returned when additional events exist.\n\n## Rate limits\n\nClients may send 120 requests per minute per API key. When the server returns **HTTP 429**, read the `Retry-After` header and delay the next attempt by the indicated number of seconds.\nDo not blindly retry without backoff.\n\n## Authentication\n\nSend `Authorization: Bearer <token>` over HTTPS. Tokens are scoped to the organization.\n\n## POST /v1/alerts/acknowledge\n\nUse this endpoint to acknowledge one alert by ID, with an analyst identifier and a reason.\n''',encoding='utf-8')

    txt=dest/'release_notes.txt'
    txt.write_text('''EVENT INGESTION SERVICE -- SYNTHETIC RELEASE NOTES\n\nVersion 2.4.1 (2025-07-09)\n- Fixed duplicate webhook deliveries after transient gateway timeouts.\n- Improved validation messages for malformed event payloads.\n\nVersion 2.4.0 (2025-06-25)\n- Introduced the event cursor endpoint for paginated retrieval.\n- Added audit logging for API token rotation.\n\nVersion 2.3.9 (2025-06-11)\n- Fixed a rare timestamp parsing error in UTC offsets.\n''',encoding='utf-8')

    img=Image.new('RGB',(1650,2200),'white')
    draw=ImageDraw.Draw(img)
    paths=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
           '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    font_path=next((p for p in paths if Path(p).exists()),None)
    def font(n):return ImageFont.truetype(font_path,n) if font_path else ImageFont.load_default()
    draw.text((110,110),'SYNTHETIC INVOICE - OCR TEST', fill='black',font=font(54))
    draw.line((110,200,1500,200),fill='black',width=4)
    lines=[('Invoice ID: INV-2025-417',310),('Issue date: 14 August 2025',420),
           ('Bill to: Demo Customer Ltd',530),('Description             Amount INR',800),
           ('Network monitoring          12,000',925),('Incident support             6,450',1030),
           ('TOTAL DUE (INR): 18,450',1230),('Payment due: 28 August 2025',1390),
           ('For benchmark testing only - not a real invoice',1670)]
    for s,y in lines: draw.text((115,y),s,fill='black',font=font(46 if y<1500 else 35))
    pdf=dest/'scanned_invoice.pdf'
    img.save(pdf,'PDF',resolution=150.0)
    return [docx,md,txt,pdf]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',default='.',help='Write the four fixtures into this directory')
    args=p.parse_args()
    for path in generate(args.out_dir): print(path)
