"""Run a labeled retrieval/answer benchmark against the existing DocumentRAG.

No framework or model substitution: imports the user's rag.py directly.
Default is retrieval-only (zero Groq calls). Use --mode full for live answers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def normalize(value: object) -> str:
    return ''.join(re.findall(r'[a-z0-9]+', str(value).casefold()))


def contains(value: object, phrase: str) -> bool:
    """Loose punctuation/case normalization, strict numeric token boundaries."""
    s = str(value)
    term = phrase.strip()
    if re.fullmatch(r'\d[\d,]*(?:\.\d+)?', term):
        target = term.replace(',', '')
        numbers = re.findall(r'(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])', s)
        return target in (n.replace(',', '') for n in numbers)
    return bool(term) and normalize(term) in normalize(s)


def hit_matches_anchor(hit: dict[str, Any], anchor: dict[str, Any]) -> bool:
    if hit.get('filename', '').casefold() != anchor.get('filename', '').casefold():
        return False
    if 'page' in anchor and hit.get('page') != anchor['page']:
        return False
    return all(contains(hit.get('text', ''), term) for term in anchor.get('contains', []))


def match_groups(hits: list[dict[str, Any]], groups: list[list[dict[str, Any]]]) -> tuple[set[int], set[int]]:
    """Return matched gold group indices and matched *retrieved source* indices."""
    matched_groups, matched_hits = set(), set()
    for gi, alternatives in enumerate(groups):
        for hi, hit in enumerate(hits):
            if any(hit_matches_anchor(hit, anchor) for anchor in alternatives):
                matched_groups.add(gi)
                matched_hits.add(hi)
    return matched_groups, matched_hits


def extract_citations(answer: str) -> list[int]:
    patterns = [r'\[\s*S\s*(\d+)\s*\]', r'【\s*S\s*(\d+)\s*】']
    matches = []
    for pattern in patterns:
        matches.extend(int(n) for n in re.findall(pattern, answer, flags=re.I))
    return sorted(set(matches))


def answer_proxy(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    if case['kind'] == 'unanswerable':
        patterns = (r'not (?:present|available|provided|stated|mentioned|specified|included|found)',
                    r'cannot (?:determine|find|provide|be determined|be answered)',
                    r'does not (?:contain|provide|mention|specify)',
                    r'insufficient (?:evidence|information|data)',
                    r'no (?:information|data|evidence)', r'not in (?:the|this) (?:document|source)',
                    r'unable to (?:determine|find|provide)')
        refusal = any(re.search(p, answer, flags=re.I) for p in patterns)
        return {'expected_groups':0, 'groups_hit':0,'coverage':None,
                'abstention_detected':refusal, 'proxy_pass':refusal}
    targets = case.get('expected_answer', [])
    present = [any(contains(answer, alt) for alt in group) for group in targets]
    return {'expected_groups':len(present),'groups_hit':sum(present),
            'coverage':sum(present)/len(present) if present else None,
            'abstention_detected':None, 'proxy_pass':all(present) if present else None}


def score_case(case: dict[str, Any], hits: list[dict[str, Any]], answer: str | None,
               *, k: int=5) -> dict[str, Any]:
    top = hits[:k]
    groups = case['gold_evidence']
    gold_ids, supported_hits = match_groups(top, groups)
    is_answerable = case['kind'] != 'unanswerable'
    result = {
        'id': case['id'], 'corpus':case['corpus'], 'kind':case['kind'], 'question':case['question'],
        'gold_group_count':len(groups), 'gold_group_hits_at5':len(gold_ids),
        'evidence_recall_at5':len(gold_ids)/len(groups) if groups else None,
        # Conventional fixed-denominator P@5 plus adaptive purity for systems
        # designed to return fewer than 5 citations on purpose.
        'precision_at5_fixed':len(supported_hits)/k if is_answerable else None,
        'source_purity_at5':(len(supported_hits)/len(top) if top else 0.0) if is_answerable else None,
        'returned_at5':len(top), 'relevant_sources_at5':len(supported_hits),
        'sources':[{k2:h.get(k2) for k2 in ('filename','page','locator','kind','score','rerank_score','reason','evidence_type')}
                   | {'text_preview':str(h.get('text',''))[:1500]} for h in hits],
    }
    if answer is not None:
        cited = extract_citations(answer)
        valid = [n for n in cited if 1 <= n <= len(hits)]
        invalid = [n for n in cited if n not in valid]
        cited_hits = [hits[n-1] for n in valid]
        supported_gold, aligned = match_groups(cited_hits, groups)
        result.update({
            'answer':answer,
            'cited_source_ids':[f'S{n}' for n in cited],
            'invalid_source_ids':[f'S{n}' for n in invalid],
            'citation_id_validity':len(valid)/len(cited) if cited else None,
            'citation_alignment_precision':(len(aligned)/len(valid) if valid else 0.0) if groups else None,
            'citation_gold_coverage':len(supported_gold)/len(groups) if groups else None,
            # No relevant citations are expected for an unanswerable question.
            'unanswerable_cites_unrelated_sources':bool(cited) if not groups else None,
            'answer_proxy':answer_proxy(answer,case),
        })
    return result


def mean_field(items:list[dict[str,Any]],field:str)->float|None:
    vals=[float(item[field]) for item in items if item.get(field) is not None]
    return round(statistics.mean(vals),4) if vals else None


def summarize(rows:list[dict[str,Any]],mode:str,judgments:dict[str,dict[str,Any]]|None=None)->dict[str,Any]:
    valid=[x for x in rows if 'error' not in x]
    answerable=[x for x in valid if x['kind']!='unanswerable']
    negatives=[x for x in valid if x['kind']=='unanswerable']
    res={
        'cases_attempted':len(rows),'cases_completed':len(valid),'errors':len(rows)-len(valid),
        'answerable_cases':len(answerable), 'unanswerable_cases':len(negatives),
        'macro_evidence_recall_at5':mean_field(answerable,'evidence_recall_at5'),
        'macro_precision_at5_fixed':mean_field(answerable,'precision_at5_fixed'),
        'macro_source_purity_at5':mean_field(answerable,'source_purity_at5'),
        'full_evidence_coverage_rate':round(sum(r['evidence_recall_at5']==1 for r in answerable)/len(answerable),4) if answerable else None,
        'mean_sources_returned_on_unanswerable':round(statistics.mean(r['returned_at5'] for r in negatives),2) if negatives else None,
        'median_retrieval_seconds':round(statistics.median([x['retrieval_seconds'] for x in valid if isinstance(x.get('retrieval_seconds'),(int,float))]),3) if any(isinstance(x.get('retrieval_seconds'),(int,float)) for x in valid) else None,
    }
    if mode=='full':
        res.update({
            'macro_citation_id_validity':mean_field(valid,'citation_id_validity'),
            'macro_citation_alignment_precision':mean_field(answerable,'citation_alignment_precision'),
            'macro_citation_gold_coverage':mean_field(answerable,'citation_gold_coverage'),
            'answer_citation_presence_rate':round(sum(bool(x.get('cited_source_ids')) for x in answerable)/len(answerable),4) if answerable else None,
            'automatic_answer_key_coverage':mean_field(answerable,'_proxy_coverage'),
            'automatic_answer_key_all_pass_rate':round(sum(x.get('answer_proxy',{}).get('proxy_pass') is True for x in answerable)/len(answerable),4) if answerable else None,
            'unanswerable_abstention_rate':round(sum(x.get('answer_proxy',{}).get('abstention_detected') is True for x in negatives)/len(negatives),4) if negatives else None,
            'unanswerable_no_citation_rate':round(sum(not x.get('cited_source_ids') for x in negatives)/len(negatives),4) if negatives else None,
        })
    if judgments:
        rated=[(r,judgments[r['id']]) for r in valid if r['id'] in judgments]
        def pct(field):
            vals=[j[field] for r,j in rated if j.get(field) in (0,1)]
            return round(sum(vals)/len(vals),4) if vals else None
        res['human_reviewed_count']=len(rated)
        res['human_answer_accuracy']=pct('answer_correct')
        res['human_citation_correctness']=pct('citations_correct')
    return res


def parse_judgments(path:Path)->dict[str,dict[str,Any]]:
    if not path.is_file():return {}
    rows={}
    with path.open('r',encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if not row.get('id'):continue
            out={}
            for field in ('answer_correct','citations_correct'):
                v=row.get(field,'').strip()
                out[field]=int(v) if v in ('0','1') else None
            rows[row['id']]=out
    return rows


def save_review(path:Path,rows:list[dict[str,Any]],prior:dict[str,dict[str,Any]])->None:
    cols=['id','corpus','kind','question','expected_answer','answer','automatic_proxy_pass',
          'evidence_recall_at5','citation_alignment_precision','answer_correct','citations_correct','review_notes']
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        out=csv.DictWriter(handle,fieldnames=cols)
        out.writeheader()
        for row in rows:
            prior_r=prior.get(row['id'],{})
            out.writerow({
                'id':row['id'],'corpus':row['corpus'],'kind':row['kind'],
                'question':row['question'],
                'expected_answer':json.dumps(row.get('_expected_answer',[]),ensure_ascii=False),
                'answer':row.get('answer',''),
                'automatic_proxy_pass':row.get('answer_proxy',{}).get('proxy_pass',''),
                'evidence_recall_at5':row.get('evidence_recall_at5',''),
                'citation_alignment_precision':row.get('citation_alignment_precision',''),
                'answer_correct':prior_r.get('answer_correct') if prior_r.get('answer_correct') is not None else '',
                'citations_correct':prior_r.get('citations_correct') if prior_r.get('citations_correct') is not None else '',
                'review_notes':''})


def print_summary(name:str,summary:dict[str,Any]):
    print('\n=== '+name+' ===')
    for key in ('cases_completed','errors','macro_evidence_recall_at5',
                'macro_precision_at5_fixed','macro_source_purity_at5',
                'full_evidence_coverage_rate','mean_sources_returned_on_unanswerable','median_retrieval_seconds',
                'automatic_answer_key_coverage','automatic_answer_key_all_pass_rate',
                'macro_citation_id_validity','macro_citation_alignment_precision',
                'macro_citation_gold_coverage','answer_citation_presence_rate','unanswerable_abstention_rate',
                'unanswerable_no_citation_rate','human_answer_accuracy','human_citation_correctness'):
        if key in summary:print(f'  {key:42} {summary[key]}')


def save_markdown_report(path:Path, report:dict[str,Any])->None:
    summary=report['overall']
    def fmt(v):
        if v is None:return 'N/A'
        if isinstance(v,float):return f'{v*100:.1f}%' if 0 <= v <= 1 else f'{v:.3f}'
        return str(v)
    rows=[
        '# Document Intelligence RAG — Benchmark Results',
        '',
        f"**Generated:** {report.get('generated_at','unknown')}  ",
        f"**Mode:** {report['mode']}  ",
        f"**Cases completed:** {summary['cases_completed']} of {summary['cases_attempted']}  ",
        '**Important:** Evidence Recall@5 is labeled-evidence-group coverage, '
        'not exhaustive corpus recall. Automatic answer checks are lexical proxies.',
        '',
        '## Overall retrieval quality', '',
        '| Metric | Value |', '|---|---:|'
    ]
    for label,key in [
        ('Evidence Recall@5','macro_evidence_recall_at5'),
        ('Precision@5 (fixed denominator)','macro_precision_at5_fixed'),
        ('Source Purity@5','macro_source_purity_at5'),
        ('Complete evidence coverage','full_evidence_coverage_rate'),
        ('Median retrieval latency (seconds)','median_retrieval_seconds')]:
        v=summary.get(key)
        rows.append(f'| {label} | {fmt(v) if key != "median_retrieval_seconds" else (str(v) if v is not None else "N/A")} |')
    rows+=['','## By document corpus','',
           '| Corpus | Cases | Evidence Recall@5 | Precision@5 | Source Purity@5 |',
           '|---|---:|---:|---:|---:|']
    for name,d in report.get('by_corpus',{}).items():
        rows.append(f"| {name} | {d['cases_completed']} | {fmt(d.get('macro_evidence_recall_at5'))} | "
                    f"{fmt(d.get('macro_precision_at5_fixed'))} | {fmt(d.get('macro_source_purity_at5'))} |")
    if report['mode']=='full':
        rows+=['','## Answer and citation checks','',
               '| Metric | Value |', '|---|---:|']
        for label,key in [
            ('Automatic answer-key coverage (heuristic)','automatic_answer_key_coverage'),
            ('Citation ID validity','macro_citation_id_validity'),
            ('Citation alignment (passage proxy)','macro_citation_alignment_precision'),
            ('Citation gold coverage','macro_citation_gold_coverage'),
            ('Answerable question citation presence','answer_citation_presence_rate'),
            ('Unanswerable question abstention','unanswerable_abstention_rate'),
            ('Human-reviewed answer accuracy','human_answer_accuracy'),
            ('Human-reviewed citation correctness','human_citation_correctness')]:
            rows.append(f'| {label} | {fmt(summary.get(key))} |')
    misses=[r for r in report['results'] if r.get('error') or
            (r.get('kind')!='unanswerable' and r.get('evidence_recall_at5',1)<1)]
    rows+=['','## Retrieval failures requiring inspection','',
           '| ID | Corpus | Recall@5 | Question / error |', '|---|---|---:|---|']
    if not misses:rows.append('| — | — | — | No failed evidence checks in this run |')
    for r in misses:
        text=str(r.get('error') or r.get('question','')).replace('|', '\\|').replace('\n',' ')
        if len(text)>180:text=text[:177]+'...'
        rows.append(f"| {r['id']} | {r['corpus']} | {fmt(r.get('evidence_recall_at5'))} | {text} |")
    rows+=['','## Interpretation','',
           '- Verify false positives and missed gold markers before making model claims.',
           '- A correct citation ID does not mean the cited claim is factually supported.',
           '- Review 0/1 judgments in the generated CSV for real answer accuracy.',
           '- Keep the holdout split separate until you finish tuning the pipeline.',
           '- Record the document versions, model/device settings and git commit.', '']
    path.write_text('\n'.join(rows),encoding='utf-8')


def validate_cases(data:dict[str,Any])->None:
    if data.get('schema_version')!=1:raise ValueError('Unsupported benchmark_cases.json schema')
    seen=set()
    for case in data['cases']:
        if case['id'] in seen:raise ValueError('Duplicate case ID: '+case['id'])
        seen.add(case['id'])
        if case['corpus'] not in data['corpora']:raise ValueError('Unknown corpus in '+case['id'])
        if case['kind'] not in ('factual','unanswerable'):raise ValueError('Unknown kind in '+case['id'])
        if case['kind']=='unanswerable' and case['gold_evidence']:
            raise ValueError('Unanswerable case must not have gold evidence: '+case['id'])
        if case['kind']=='factual' and not case['gold_evidence']:
            raise ValueError('Factual case requires gold evidence: '+case['id'])
        for group in case['gold_evidence']:
            if not group:raise ValueError('Empty OR group: '+case['id'])
            for anchor in group:
                if anchor['filename'] not in data['corpora'][case['corpus']]['documents']:
                    raise ValueError('Gold anchor file not in corpus: '+case['id'])
                if not anchor.get('contains'):raise ValueError('Gold anchor lacks content markers: '+case['id'])


def main()->int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir',type=Path,default=Path('.'),help='Project directory with sample.pdf, report.pdf and fixtures')
    p.add_argument('--cases',type=Path,default=HERE/'benchmark_cases.json')
    p.add_argument('--mode',choices=('retrieval','full'),default='retrieval',help='Full mode calls Groq for every question')
    p.add_argument('--corpus',choices=('all','syllabus','report','fixtures'),default='all')
    p.add_argument('--split',choices=('all','dev','holdout'),default='all',help='Keep held-out examples untouched during tuning')
    p.add_argument('--only-id',action='append',help='Evaluate selected case ID (repeat as needed)')
    p.add_argument('--limit',type=int,default=None,help='Maximum questions, after filtering')
    p.add_argument('--top-k',type=int,default=8,help='Retrieve this many passages; scoring always uses first 5')
    p.add_argument('--ocr',choices=('off','auto','all'),default='auto')
    p.add_argument('--device',default=None,help='cuda, cpu, or auto (default: RAG_DEVICE from .env)')
    p.add_argument('--no-rerank',action='store_true')
    p.add_argument('--reindex',action='store_true')
    p.add_argument('--skip-missing',action='store_true',help='Explicitly skip corpora with missing files')
    p.add_argument('--out',type=Path,default=Path('benchmark_results.json'))
    p.add_argument('--judgments',type=Path,default=None,help='Previously human-reviewed CSV with two 0/1 columns')
    p.add_argument('--rescore-report',type=Path,default=None,help='Recalculate human accuracy from saved JSON and reviewed CSV without Groq calls')
    p.add_argument('--self-test',action='store_true',help='No-model validation of benchmark metric calculations')
    args=p.parse_args()
    if args.self_test:
        demo={'id':'demo','corpus':'demo','kind':'factual','question':'demo',
              'gold_evidence':[[{'filename':'x.pdf','page':2,'contains':['widget','100']}]],
              'expected_answer':[['100']]}
        hits=[{'filename':'x.pdf','page':2,'text':'widget price 100'},
              {'filename':'x.pdf','page':3,'text':'wrong'}]
        s=score_case(demo,hits,'The widget costs 100 [S1].')
        assert s['evidence_recall_at5']==1 and s['source_purity_at5']==.5
        assert s['precision_at5_fixed']==.2
        assert s['citation_alignment_precision']==1
        assert s['answer_proxy']['proxy_pass']
        assert contains('Revenue 1,36,592 crore','136592')
        assert not contains('163592','136592')
        assert extract_citations('Test 【S2】 and [S4].')==[2,4]
        print('BENCHMARK METRIC SELF-TEST PASSED')
        return 0
    if args.rescore_report:
        if not args.judgments:
            p.error('--rescore-report requires --judgments')
        prior = parse_judgments(args.judgments.expanduser().resolve())
        saved = args.rescore_report.expanduser().resolve()
        report = json.loads(saved.read_text(encoding='utf-8'))
        rows = report['results']
        mode = report['mode']
        report['overall'] = summarize(rows,mode,prior)
        report['by_corpus'] = {corpus:summarize([r for r in rows if r['corpus']==corpus],mode,prior)
                               for corpus in dict.fromkeys(r['corpus'] for r in rows)}
        target = saved.with_name(saved.stem+'.graded.json')
        target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        save_markdown_report(target.with_suffix('.md'), report)
        print_summary('HUMAN-REVIEWED BENCHMARK',report['overall'])
        print('Saved graded report: '+str(target))
        return 0
    data=json.loads(args.cases.read_text(encoding='utf-8'))
    validate_cases(data)
    selected=[c for c in data['cases'] if
              (args.corpus=='all' or c['corpus']==args.corpus) and
              (args.split=='all' or c.get('split')==args.split) and
              (not args.only_id or c['id'] in args.only_id)]
    if args.limit is not None:
        if args.limit<1:p.error('--limit must be positive')
        selected=selected[:args.limit]
    if args.top_k<5:p.error('--top-k must be >= 5 to measure @5')
    if not selected:p.error('No cases selected')
    data_dir=args.data_dir.expanduser().resolve()
    corpora=list(dict.fromkeys(c['corpus'] for c in selected))
    availability={}
    for corpus in corpora:
        missing=[name for name in data['corpora'][corpus]['documents'] if not (data_dir/name).is_file()]
        availability[corpus]=not missing
        if missing:
            msg=f'Missing {corpus} files in {data_dir}: '+', '.join(missing)
            if args.skip_missing:print('[SKIP] '+msg)
            else:p.error(msg+'; run benchmark_fixtures.py for synthetic files or specify --skip-missing')
    selected=[c for c in selected if availability[c['corpus']]]
    if not selected:
        print('No corpus available, nothing to evaluate.')
        return 2
    # Import the user's local engine by source directory, not the benchmark's.
    sys.path.insert(0,str(data_dir))
    try:
        from rag import DocumentRAG
    except ImportError as exc:
        p.error(f'Cannot import rag.py from {data_dir}: {exc}')
    prior=parse_judgments(args.judgments.expanduser().resolve()) if args.judgments else {}
    results=[]
    for corpus in dict.fromkeys(c['corpus'] for c in selected):
        files=[(name,(data_dir/name).read_bytes()) for name in data['corpora'][corpus]['documents']]
        print(f'\n=== INDEX {corpus}: '+', '.join(name for name,_ in files)+' ===',flush=True)
        engine=DocumentRAG(cache_dir=data_dir/'.rag_cache',device=args.device,
                           ocr=args.ocr,rerank=not args.no_rerank)
        engine.build_index(files,refresh=args.reindex)
        for case in [c for c in selected if c['corpus']==corpus]:
            t0=time.perf_counter()
            try:
                if args.mode=='full':
                    response=engine.ask(case['question'],top_k=args.top_k)
                    answer=response['answer']
                    hits=response['sources']
                else:
                    answer=None
                    hits=engine.retrieve(case['question'],top_k=args.top_k)
                secs=round(time.perf_counter()-t0,3)
                scored=score_case(case,hits,answer)
                scored['retrieval_seconds']=secs if args.mode=='retrieval' else None
                scored['total_question_seconds']=secs
                if args.mode=='full':
                    scored['_proxy_coverage']=scored.get('answer_proxy',{}).get('coverage')
                scored['_expected_answer']=case.get('expected_answer',[])
                results.append(scored)
                recall=scored['evidence_recall_at5']
                recall_text='N/A' if recall is None else f'{recall:.2f}'
                print(f"{case['id']}: gold-recall@5={recall_text}  "
                      f"sources={len(hits)}  time={secs:.2f}s"+
                      ('  ANSWER-PROXY='+str(scored['answer_proxy']['proxy_pass']) if answer is not None else ''),flush=True)
            except Exception as exc:
                print(f'{case["id"]}: ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
                results.append({'id':case['id'],'corpus':corpus,'kind':case['kind'],
                                'question':case['question'], 'error':f'{type(exc).__name__}: {exc}'})
    all_summary=summarize(results,args.mode,prior)
    by_corpus={c:summarize([r for r in results if r['corpus']==c],args.mode,prior)
               for c in corpora if any(r['corpus']==c for r in results)}
    output={
        'benchmark':'Document Intelligence RAG Benchmark v1',
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'mode':args.mode,'configuration':{'corpus':args.corpus,'split':args.split,'top_k':args.top_k,
                'scored_at_k':5,'ocr':args.ocr,'device':engine.device,
                'reranker_enabled':not args.no_rerank,'question_count':len(selected)},
        'metric_definitions':{
            'evidence_recall_at5':'Fraction of manually annotated required evidence groups supported by top five returned source excerpts. Each group can have multiple acceptable source anchors.',
            'precision_at5_fixed':'Number of gold-matching source excerpts in top five divided by 5 (non-returned slots count as not relevant).',
            'source_purity_at5':'Number of gold-matching source excerpts divided by number actually returned (max 5); suitable for variable-length evidence lists.',
            'citation_alignment_precision':'Fraction of cited source IDs whose excerpts match at least one gold evidence group; NOT sentence-level claim verification.',
            'citation_gold_coverage':'Fraction of gold groups reached by at least one cited passage.',
            'automatic_answer_key_coverage':'Fraction of expected literal answer groups present; a heuristic, not semantic factual correctness.',
            'human_answer_accuracy':'Fraction of manually adjudicated full answers labeled correct (0/1) in judgments CSV.',
        },'overall':all_summary,'by_corpus':by_corpus,'results':results}
    out=args.out.expanduser().resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    save_markdown_report(out.with_suffix('.md'),output)
    print_summary('OVERALL',all_summary)
    for c,r in by_corpus.items():print_summary(c.upper(),r)
    print(f'\nSaved JSON report: {out}')
    print('Saved Markdown report: '+str(out.with_suffix('.md')))
    if args.mode=='full':
        review=out.with_suffix('.review.csv')
        save_review(review,results,prior)
        print('Manual review template: '+str(review))
        print('IMPORTANT: automatic answer checks are lexical proxies; review answers/citations manually for real accuracy.')
    if args.mode=='retrieval':print('No Groq calls made. Use --mode full --limit N for real answer/citation evaluation.')
    return 1 if all_summary['errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
