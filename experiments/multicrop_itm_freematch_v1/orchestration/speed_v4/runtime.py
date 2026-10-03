"""Admitted speed extension over immutable FreeMatch science; per-run exact gates."""
import argparse,gc,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(Path(__file__).parent))
import torch,train
from common import read,digest,atomic_json,atomic_torch,exclusive,now
from candidates import SpeedRunner
from bench import compare,compare_records
OriginalRunner=train.Runner
REG=ROOT/'orchestration/speed_v4/registration.json'
def registration():
 p=train.profile(True);r=read(REG)
 if r['original_profile_sha256']!=digest(ROOT/'configs/execution_profile.json') or r['max_workers'] not in (1,2) or r['target_steps']!=2600:raise RuntimeError('Registered speed scope differs')
 for n,h in r['source'].items():
  if digest(ROOT/n)!=h:raise RuntimeError('Independent speed extension source differs '+n)
 cpu=read(ROOT/'audit/speed_v4/cpu_preflight.json');bench=read(ROOT/'audit/speed_v4/admission.json')
 if not cpu['passed'] or cpu['source']!=r['source'] or not bench['passed'] or bench['registration_sha256']!=digest(REG):raise RuntimeError('Real CPU/GPU extension admission absent')
 return r
def bind(rid):
 r=registration()
 if rid not in r['eligible_runs']:raise RuntimeError('Only untouched registered remaining groups')
 train.bind_zero(train.base,train.profile(True))
 prior=train.base.provenance
 extension={'name':'freematch_source_preserving_GPU_speed_v4','registration_sha256':digest(REG),'variant':r['variant'],'actual_activation_checkpointing':r['variant'] in ('reference','retained_checkpoint'),'strong_tail_extra_forward_visits':83200 if r['variant']=='reference' else 0,'max_concurrent_workers':r['max_workers'],'original_profile_sha256':r['original_profile_sha256']}
 def provenance(cfg,manifest):return {**prior(cfg,manifest),'execution_extension':extension}
 train.base.provenance=provenance
 Parent=OriginalRunner if r['variant']=='reference' else SpeedRunner
 class ProductionRunner(Parent):
  def __init__(self,cfg,contract,prov):
   if r['variant']=='reference':super().__init__(cfg,contract,prov)
   else:super().__init__(cfg,contract,prov,checkpointing=r['variant']=='retained_checkpoint',decoded_capacity=256)
 train.Runner=train.base.Runner=ProductionRunner
 return r
def gate(rid):
 r=bind(rid);train.gate(rid);gc.collect();torch.cuda.empty_cache()
 cfg,contract,prov,_=train.check(rid,True);out=ROOT/'audit/gpu_gates'/rid
 # Original reference is independently rebuilt; no copied cache and no formal steps.
 reference=OriginalRunner(cfg,contract,prov);a=reference.train_step();b=reference.train_step();v=reference.snapshot()
 v['provenance']={k:x for k,x in prov.items() if k!='execution_extension'}
 v['engineering_execution']=dict(mode='original_science_checkpointed_two_pass_reference',actual_activation_checkpointing=True,formal_steps_added=0,original_profile_sha256=r['original_profile_sha256'],engineering_source=r['source'])
 atomic_torch(out/'original_science_reference2.pt',v)
 candidate=torch.load(out/'reference2.pt',map_location='cpu',weights_only=False);compare(v,candidate)
 cold=read(out/'passed.json');compare_records(b,cold['two'])
 atomic_json(ROOT/'audit/speed_v4/per_run_admission'/(rid+'.json'),dict(passed=True,registration_sha256=digest(REG),original_numerical_reference_and_full_cold_replay_passed=True,no_Validation_Test=True,formal_steps_added=0,gate_receipt_sha256=digest(out/'passed.json'),original_reference_sha256=digest(out/'original_science_reference2.pt'),created_utc=now()))
def verify(rid):
 result=read(ROOT/'outputs'/rid/'result.json')
 if result['provenance'].get('execution_extension'):bind(rid)
 proof=train.completed(rid);proof['independent_speed_version']=result['provenance'].get('execution_extension')
 atomic_json(ROOT/'audit/completed'/(rid+'.json'),proof);print(json.dumps(dict(passed=True,run_id=rid)),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--run-id',required=True);a.add_argument('--slot',type=int,default=0);g=a.add_mutually_exclusive_group(required=True)
 for name in ('check','gate','run','verify-result'):g.add_argument('--'+name,action='store_true')
 args=a.parse_args();rid=args.run_id
 if args.verify_result:return verify(rid)
 if args.check:
  r=bind(rid);_,_,_,step=train.check(rid,True)
  if step is not None:raise RuntimeError('No implicit formal resume')
  return print(json.dumps(dict(passed=True,run_id=rid)),flush=True)
 if torch.cuda.device_count()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=train.GPU:raise RuntimeError('Registered GPU identity required')
 if args.gate:
  with exclusive(ROOT/'locks'/(train.GPU+'.lock')),exclusive(ROOT/'locks'/(rid+'.lock')):gate(rid)
 else:
  import fcntl
  r=bind(rid)
  if not 0<=args.slot<r['max_workers']:raise RuntimeError('Outside admitted slot capacity')
  proof=read(ROOT/'audit/speed_v4/per_run_admission'/(rid+'.json'))
  if not proof['passed'] or proof['registration_sha256']!=digest(REG) or proof['gate_receipt_sha256']!=digest(ROOT/'audit/gpu_gates'/rid/'passed.json'):raise RuntimeError('Actual own-run reference+cold gate absent')
  out=ROOT/'outputs'/rid
  if any((out/n).exists() for n in ('last.pt','best.pt','status.json','result.json')) or list(out.glob('train_*.jsonl')):raise RuntimeError('Existing formal evidence; separate recovery authority required')
  # Original EX lock now remains mutually exclusive with shared admitted speed workers.
  with (ROOT/'locks'/(train.GPU+'.lock')).open('rb',buffering=0) as card:
   fcntl.flock(card,fcntl.LOCK_SH|fcntl.LOCK_NB)
   try:
    with exclusive(ROOT/'locks'/('speed_slot%d.lock'%args.slot)),exclusive(ROOT/'locks'/(rid+'.lock')):train.base.run(rid,train.GPU)
   finally:fcntl.flock(card,fcntl.LOCK_UN)
if __name__=='__main__':main()
