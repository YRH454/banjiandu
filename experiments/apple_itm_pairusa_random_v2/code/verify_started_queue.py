"""Read durable state to certify actual startup, not just an existing process."""
import argparse
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--minimum-step',type=int,default=50);a=p.parse_args()
    out=ROOT/'outputs'
    smoke=read(out/'smoke_pipeline_state.json')
    assert smoke['state']=='completed_validation'
    smoke_a=read(out/'apple_itm_random_A_bce_l005_s20260825_smoke20/result.json')
    smoke_b=read(out/'apple_itm_random_B_bce_pairusa_l005_s20260825_smoke20/result.json')
    assert smoke_a['initial_shared_sha256']==smoke_b['initial_shared_sha256']
    b_logs=[json.loads(line) for line in (out/'apple_itm_random_B_bce_pairusa_l005_s20260825_smoke20/train.jsonl').read_text().splitlines()]
    assert len(b_logs)==20 and all(math.isfinite(r['loss']) for r in b_logs)
    assert any(r['pairusa']>0 and r['lambda']>0 for r in b_logs)
    state=read(out/'pipeline_state.json')
    run_dir=out/state['active_run']
    assert '_smoke' not in run_dir.name
    last=run_dir/'last.pt'
    if not last.exists():print('PENDING: no durable formal checkpoint yet');return
    ckpt=torch.load(last,map_location='cpu',weights_only=False)
    if ckpt['step']<a.minimum_step:print(f"PENDING: durable checkpoint step={ckpt['step']}");return
    assert ckpt['provenance']['base']['source']==state['source_fingerprint']
    assert ckpt['provenance']['base']['run']['training']['max_steps']==1600
    assert ckpt['provenance']['base']['run']['training']['pairusa_start']==100
    for filename,expected in state['source_fingerprint'].items():
        if filename=='tokenizer_snapshot':continue
        path=Path(filename)
        if not path.is_absolute():path=ROOT/path
        assert sha(path)==expected,f'Changed registered file: {path}'
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
            'real_data_smoke_a_steps':20,'real_data_smoke_b_steps':20,
            'pairusa_exercised_on_real_data':True,'a_b_initial_shared_sha256':smoke_a['initial_shared_sha256'],
            'active_formal_run':state['active_run'],'durable_formal_checkpoint_step':ckpt['step'],
            'formal_steps_target':1600,'checkpoint_loadable':True,'registered_source_hashes_match':True,
            'last_checkpoint_path':str(last),'last_checkpoint_sha256_at_verification':sha(last),
            'semantic_negative_status':'random_other_case_unverified','independent_test_evaluation':False}
    (ROOT/'audit/verified_formal_start.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
