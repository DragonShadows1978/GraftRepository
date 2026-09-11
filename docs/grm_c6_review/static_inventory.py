#!/usr/bin/env python3
# Prior art: Python Software Foundation, Python 3.12 (2023); pytest team,
# pytest 9.0.2 (installed 2026 audit snapshot). Reused AST/receipt plumbing,
# no new runtime algorithm. Executed pre-annotation bytes retained by SHA.
"""Prior art: Python Software Foundation AST visitor; ordinary static inventory.
No new runtime algorithm. Literal path extraction is conservative, not a proof
that a string is executed. Dynamic expressions remain explicitly unresolved.
"""
import ast, csv, hashlib, json, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
names=[x.split('\t')[-1] for x in (OUT/'branch_names.tsv').read_text().splitlines()]
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
trees={b:set(git('ls-tree','-r','--name-only',b).decode().splitlines()) for b in ('main','lc1-wip')}
def source(b,p):
    return git('show',b+':'+p).decode() if p in trees[b] else ''
def inventory(src):
    if not src: return {}
    tree=ast.parse(src); out={}
    def record(k,n,v): out[k]=(n.lineno,ast.unparse(v) if isinstance(v,ast.AST) else str(v))
    class Visitor(ast.NodeVisitor):
        def __init__(self): self.scope=[]
        def visit_ClassDef(self,n):
            self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
        def visit_FunctionDef(self,n):
            self.scope.append(n.name)
            pos=n.args.posonlyargs+n.args.args
            for arg,val in zip(pos[-len(n.args.defaults):] if n.args.defaults else [], n.args.defaults): record('parameter:'+'.'.join(self.scope)+':'+arg.arg,n,val)
            for arg,val in zip(n.args.kwonlyargs,n.args.kw_defaults):
                if val is not None: record('parameter:'+'.'.join(self.scope)+':'+arg.arg,n,val)
            self.generic_visit(n);self.scope.pop()
        visit_AsyncFunctionDef=visit_FunctionDef
        def visit_Assign(self,n):
            for t in n.targets:
                label=ast.unparse(t)
                if label.isupper() or 'environ[' in label: record('assignment:'+'.'.join(self.scope)+':'+label,n,n.value)
            self.generic_visit(n)
        def visit_AnnAssign(self,n):
            label=ast.unparse(n.target)
            if label.isupper() and n.value is not None: record('assignment:'+'.'.join(self.scope)+':'+label,n,n.value)
            self.generic_visit(n)
        def visit_Call(self,n):
            label=ast.unparse(n.func)
            if label.endswith('.add_argument'):
                flags=','.join(ast.unparse(a) for a in n.args)
                kws={k.arg:k.value for k in n.keywords}
                value='; '.join(k+'='+ast.unparse(v) for k,v in kws.items() if k in ('default','action','required','choices','nargs','const')) or 'implicit argparse default None'
                record('cli:'+'.'.join(self.scope)+':'+flags,n,value)
            if ('environ' in label or 'getenv' in label or label=='env.get') and n.args:
                record('env:'+'.'.join(self.scope)+':'+ast.unparse(n.args[0]),n,n)
            self.generic_visit(n)
    Visitor().visit(tree);return out
rows=[]; refs=[]
for p in names:
    if not p.endswith(('.py','.md','.json','.sh')): continue
    old,new=source('main',p),source('lc1-wip',p)
    if p.endswith('.py'):
        a,b=inventory(old),inventory(new)
        for key in sorted(a.keys()|b.keys()):
            av=a.get(key,(0,'ABSENT'));bv=b.get(key,(0,'ABSENT'))
            if av[1]!=bv[1]: rows.append([p,key,*av,*bv,'New harness/internal default: not a production adoption receipt; see file order/registration and reference inventory' if p.startswith(('scripts/','tests/')) else 'See review production default table; constants/internal defaults do not independently certify behavior'])
    for i,line in enumerate(new.splitlines(),1):
        for match in re.finditer(r'(?:/mnt/(?:ForgeRealm|Shared)/|/home/vader/|/tmp/|(?:\.\./)?(?:artifacts|orders|docs|config|core|scripts|tests)/)[\w./{}+@=:-]+',line):
            raw=match.group(0).rstrip('.:'); rel=raw[3:] if raw.startswith('../') else raw
            absolute=raw.startswith('/'); local=Path(raw) if absolute else ROOT/rel
            tracked_main=rel in trees['main'];tracked_lc=rel in trees['lc1-wip']
            refs.append([p,i,raw,local.exists(),tracked_main,tracked_lc, ('absolute-external' if absolute else 'tracked-new-in-lc1' if tracked_lc and not tracked_main else 'tracked-both' if tracked_main and tracked_lc else 'untracked-or-dynamic'), (Path('/mnt/ForgeRealm/GraftRepository')/rel).exists() if not absolute else local.exists()])
with (OUT/'defaults_inventory.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['file','key','main_line','main_value','lc1_line','lc1_value','evidence_class']);w.writerows(rows)
with (OUT/'references_inventory.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['source','line','reference','exists_worktree','tracked_main','tracked_lc1','classification','exists_canonical']);w.writerows(refs)
(OUT/'tracked_files.json').write_text(json.dumps({b:sorted(v) for b,v in trees.items()},indent=2)+'\n')
summary={'changed_files':len(names),'default_records':len(rows),'reference_records':len(refs),'production_default_records':sum(r[0].startswith('core/') for r in rows),'inventory_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'static_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
