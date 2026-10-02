#!/usr/bin/env python3
"""Generate a browsable inventory without changing research files."""
from pathlib import Path
import html

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ('Code', 'Scripts and build files', 'Documents', 'Figures', 'Inputs', 'Results and other artifacts')

def category(p):
    if p.suffix in {'.cpp', '.hpp', '.h', '.cu'}: return GROUPS[0]
    if p.suffix in {'.py', '.sh'} or p.name.lower() == 'makefile': return GROUPS[1]
    if p.suffix in {'.md', '.tex', '.pdf', '.bib'}: return GROUPS[2]
    if p.suffix in {'.png', '.svg', '.jpg'}: return GROUPS[3]
    if p.suffix == '.mesh': return GROUPS[4]
    return GROUPS[5]

files = sorted(p for p in ROOT.rglob('*') if p.is_file()
               and not any(part.startswith('.') or part in {'__pycache__', 'archive'} for part in p.relative_to(ROOT).parts)
               and p.name != 'CATALOG.html')
from urllib.parse import quote
parts = ['<!doctype html><meta charset="utf-8"><title>Treecode workspace</title>',
         '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}li{margin:6px 0}small{color:#666}input{padding:10px;width:90%}</style>',
         '<h1>Treecode workspace</h1><p>Generated with <code>python3 tools/catalog.py</code>. Archive and hidden files are excluded. Categories are based on file extensions; PDFs include both reports and plots.</p>',
         '<input id="filter" placeholder="Filter by filename or folder" aria-label="Filter files">']
for group in GROUPS:
    members = [p for p in files if category(p) == group]
    parts.append(f'<section><h2>{group} ({len(members)})</h2><ul>')
    for p in members:
        rel = p.relative_to(ROOT).as_posix()
        parts.append(f'<li><a href="{quote(rel)}">{html.escape(rel)}</a> <small>{p.stat().st_size:,} bytes</small></li>')
    parts.append('</ul></section>')
parts.append('''<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('li').forEach(li=>li.hidden=!li.textContent.toLowerCase().includes(q));});</script>''')
(ROOT / 'CATALOG.html').write_text('\n'.join(parts), encoding='utf-8')
print(f'Cataloged {len(files)} files')
