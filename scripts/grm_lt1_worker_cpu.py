"""CPU-only process harness for the shared LT1 worker body.
Prior art: LT1 r1 prose fake / C7 a3 constructor double (GRM, 2026), reused.
No prior art known to me for this exact worker test composition.
"""
import sys
from pathlib import Path
import pytest
from scripts import grm_lt1 as lt
from scripts import grm_lt1_worker as worker
from scripts.grm_lt1_cpu import visible_answer
from scripts.grm_c7_diagnose import repository,Model


def main():
    run=Path(sys.argv[1]);r=lt.verify();cell=next(c for c in r['cells'] if c['id']==sys.argv[2])
    with pytest.MonkeyPatch.context() as patch:
        def loader(session,flags,state):
            repo=repository(session/'repository',patch);a=repo.arena
            a.width=flags['arena_width'];a.recency_mounts=2
            if state.get('codec_words'):
                a.m.codec.words=state['codec_words'];a.m.codec.ids={w:i for i,w in enumerate(a.m.codec.words)}
            class ProseModel(Model):
                def __call__(self,ids,kv_caches=None,position_offset=0,**kwargs):
                    if kv_caches is None:
                        text=self.codec.decode(ids[0]);visible=(self.injected if a.cur_mounts else '')+'\n'+text
                        self.fold_output=visible_answer(visible,text)
                    return super().__call__(ids,kv_caches,position_offset,**kwargs)
            a.m.__class__=ProseModel
            return repo,dict(model='CPU prose fake',gpu=False)
        value=worker.execute(cell,run/cell['id'],r,loader,run=run,fake=True)
        lt.create(run/cell['id']/'worker.json',value)
if __name__=='__main__':main()
