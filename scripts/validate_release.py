"""Validate public evidence identities, runnable source, examples and local links."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]


def main():
    required=['README.md','README.zh-TW.md','LICENSE','NOTICE.md','CITATION.cff','CHANGELOG.md',
              'CONTRIBUTING.md','CODE_OF_CONDUCT.md','SECURITY.md','site/index.html','site/zh.html',
              'site/app.js','.github/workflows/ci.yml','.github/workflows/pages.yml']
    assert all((ROOT/p).is_file() for p in required),'Missing release documentation'
    provenance=json.loads((ROOT/'reports/provenance.json').read_text())
    audit=json.loads((ROOT/'reports/package_validation.json').read_text())
    training=json.loads((ROOT/'reports/training_summary.json').read_text())
    metrics=json.loads((ROOT/'reports/evaluation_summary.json').read_text())
    assert provenance['model_sha256']==audit['model_sha256']==metrics['candidate_sha256']
    assert audit['passed'] and training['training']['examples']==980000
    assert metrics['overall_candidate']['canonical_exact']['correct']==214
    for name,key in [('train_downstream_sft.py','trainer_sha256'),('sft_common.py','common_sha256')]:
        assert hashlib.sha256((ROOT/'training'/name).read_bytes()).hexdigest()==training[key],name+' source changed'
    examples=json.loads((ROOT/'examples/index.json').read_text())
    predictions=json.loads((ROOT/'reports/predictions_20260919.json').read_text())
    assert len(examples)==8 and predictions['complete'] and len(predictions['rows'])==8
    by_id={r['id'].lower():r for r in predictions['rows']}
    for name in examples:
        e=json.loads((ROOT/'examples'/(name+'.json')).read_text())
        assert [m['role'] for m in e['messages']]==['system','user']
        assert e['reference']==by_id[name]['expected']
        if e['kind']=='structured_test':
            joined='<s><|system|>'+e['messages'][0]['content']+'\n<|user_channel|>'+e['messages'][1]['content']+'\n<|assistant_channel|>'
            assert joined==by_id[name]['input'],'Example prompt differs from actual tested input'
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src'] and value:links.append((current,value))
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
    for current in paths:
        assert current.stat().st_size<5_000_000, f'Unexpected large public-source file: {current}'
        if current.suffix=='.py':ast.parse(current.read_text(),filename=str(current))
        if current.suffix=='.json':json.loads(current.read_text())
        if current.suffix=='.md':
            links.extend((current,u) for u in re.findall(r'\]\(([^)]+)\)',current.read_text()))
        if current.suffix=='.html':Links().feed(current.read_text())
        if current.suffix in ['.json','.md','.html','.yml','.py','.js','.css']:
            content=current.read_text()
            assert not re.search(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]{3,}PRIVATE KEY-----)',content),f'Possible credential: {current}'
            assert '/root/'+'Barbet/' not in content, f'Private absolute path: {current}'
    for current,url in links:
        parsed=urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:continue
        target=(current.parent/unquote(parsed.path)).resolve()
        assert target.is_relative_to(ROOT) and target.exists(),f'Broken local link: {current.relative_to(ROOT)} -> {url}'
    print(f'Validated {len(paths)} public files, 8 example inputs, evidence identities and local links.')


if __name__=='__main__':main()
