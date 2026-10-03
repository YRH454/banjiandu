"""Only remaining registered unique configurations; max two memory-safe workers."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(Path(__file__).parent))
os.environ['CUDA_VISIBLE_DEVICES']=''
import train,dispatcher as original_dispatcher
from common import read,digest,atomic_json,now,exclusive
from runtime import registration
def identity(pid):return original_dispatcher.identity(pid)
def planned_wave(pending,r):
 if len(set(pending))!=len(pending):raise RuntimeError('Duplicate logical run allocation refused')
 if not pending:return []
 first=pending[0];wave=[first]
 if r['max_workers']==2:
  choices=[rid for rid in pending[1:] if r['RAM_upper_bytes'][rid]+r['RAM_upper_bytes'][first]<=r['memory_limit_bytes']*.85 and r['GPU_peak_bytes'][rid]+r['GPU_peak_bytes'][first]<=r['GPU_total_bytes']*.90]
  if choices:wave.append(choices[0])
 return wave
def save(s,state,**kw):
 s.update(state=state,updated_utc=now(),**kw);atomic_json(ROOT/'outputs/pipeline_state.json',s)
def spawn(s,rid,phase,slot=0):
 stamp=str(time.time_ns());so=ROOT/'logs'/('speed_'+rid+'_'+phase+'_'+stamp+'.stdout.log');se=so.with_name(so.name.replace('.stdout.','.stderr.'))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if phase in ('check','verify-result') else train.GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false')
 cmd=[sys.executable,'-B','-u','orchestration/speed_v4/runtime.py','--run-id',rid,'--'+phase,'--slot',str(slot)]
 with so.open('xb') as out,se.open('xb') as err:p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env)
 return p,dict(run_id=rid,slot=slot,phase=phase,identity=identity(p.pid),command=cmd,stdout=str(so),stderr=str(se))
def main():
 r=registration();state=dict(version='freematch_server3_exact_speed_v4',identity=identity(os.getpid()),original_profile_sha256=r['original_profile_sha256'],execution_extension_registration_sha256=digest(ROOT/'orchestration/speed_v4/registration.json'),target_steps=2600,max_workers=r['max_workers'],completed=[],workers=[],automatic_formal_recovery_authorized=False)
 with exclusive(ROOT/'locks/dispatcher_single.lock'),exclusive(ROOT/'locks/speed_dispatcher_single.lock'):
  try:
   if original_dispatcher.compute():raise RuntimeError('GPU still occupied at unique supervisor startup')
   for rid in train.RUNS:
    if (ROOT/'outputs'/rid/'result.json').exists():
     p,w=spawn(state,rid,'verify-result');p.wait()
     if p.returncode:raise RuntimeError('Existing completion full CPU check failed')
     state['completed'].append(rid)
   pending=[rid for rid in r['eligible_runs'] if rid not in state['completed']]
   for rid in pending:
    o=ROOT/'outputs'/rid
    if any((o/n).exists() for n in ('last.pt','best.pt','status.json','result.json')) or list(o.glob('train_*.jsonl')):raise RuntimeError('Untouched-only scheduling; existing evidence needs separate authority')
   while pending:
    wave=planned_wave(pending,r)
    # Fully idle between waves; all gates own-run and serial, never gate beside training.
    if original_dispatcher.compute():raise RuntimeError('Unexpected scientific GPU before fresh wave gates')
    for rid in wave:
     if not original_dispatcher.bootstrap.crop_ready(train.config_for(rid)['dataset']):raise RuntimeError('Full unchanged public images must currently pass SHA guard')
     for phase in ('check','gate'):
      p,w=spawn(state,rid,phase)
      save(state,'engineering_'+phase,active_run=rid,phase=phase,child=w['identity'],workers=[w])
      p.wait()
      if p.returncode:raise RuntimeError('Fresh '+phase+' failed; preserve all evidence, no retry')
    children=[]
    for slot,rid in enumerate(wave):
     # Wave plan unique and persisted before any formal child starts.
     atomic_json(ROOT/'audit/speed_v4/allocations'/(rid+'.json'),dict(run_id=rid,slot=slot,owner='this_unique_speed_dispatcher',dispatcher_identity=state['identity'],registration_sha256=digest(ROOT/'orchestration/speed_v4/registration.json'),created_utc=now()))
     p,w=spawn(state,rid,'run',slot);children.append((p,w))
    save(state,'formal_training',active_run=wave[0],active_runs=wave,phase='run',child=None,workers=[w for p,w in children])
    failed=False
    while any(p.poll() is None for p,w in children):
     states=[]
     for p,w in children:
      sf=ROOT/'outputs'/w['run_id']/'status.json';states.append({**w,'exit_code':p.poll(),'actual_identity':identity(p.pid),'status':read(sf) if sf.exists() else None})
      if p.poll() not in (None,0):failed=True
     save(state,'formal_training_other_healthy_worker_retained' if failed else 'formal_training',workers=states)
     time.sleep(3)
    if failed:
     for p,w in children:
      if p.returncode==0:
       v,meta=spawn(state,w['run_id'],'verify-result');v.wait()
       if v.returncode==0:state['completed'].append(w['run_id'])
     raise RuntimeError('Formal lane failed; healthy sibling allowed to finish/full CPU verify; no subsequent launch/recovery')
    for p,w in children:
     if p.returncode:raise RuntimeError('Formal worker failure; no automatic recovery')
     rid=w['run_id'];v,meta=spawn(state,rid,'verify-result');v.wait()
     if v.returncode:raise RuntimeError('Full completion CPU verification failed')
     state['completed'].append(rid);pending.remove(rid)
    save(state,'wave_fully_completed',workers=[],active_run=None,active_runs=[],phase=None)
   proofs={rid:read(ROOT/'audit/completed'/(rid+'.json')) for rid in train.RUNS}
   atomic_json(ROOT/'reports/fifth_summary.json',dict(configurations=20,target_steps=2600,all_fully_verified=True,results=proofs,original_profile_sha256=r['original_profile_sha256'],independent_speed_extension_registration_sha256=digest(ROOT/'orchestration/speed_v4/registration.json'),real_per_run_environment_and_execution_version_preserved=True,limits='Single seed; Validation-only selection; unequal compute/step budgets; pair-ITM adaptation not unchanged classification reproduction; pre-boundary original checkpoint/two-pass execution and remaining exact-replay-admitted retained-graph execution; concurrent same-GPU contention and actual per-run wall cost; no Test; no arbitrary long formal recovery authority',completed_utc=now()))
   save(state,'all20_full_completed_summary_verified',workers=[],active_run=None,active_runs=[])
  except BaseException:save(state,'failed_preserve_all_no_automatic_recovery',error=traceback.format_exc());raise
if __name__=='__main__':main()
