"""Isolated per-crop real GPU execution/numerical test, no formal or Validation/Test."""
import gc,json,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(Path(__file__).parent))
import torch,train
from common import read,atomic_json,atomic_torch,now
from candidates import SpeedRunner
def CPU_tree(x):
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return {k:CPU_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [CPU_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(CPU_tree(v) for v in x)
 return x
def compare(a,b):
 a,b=CPU_tree(a),CPU_tree(b)
 for k in ('trainable_state','ema'):train.assert_tree(a[k],b[k],rtol=0,atol=1e-6)
 for k in ('optimizer','algorithm_state'):train.assert_tree(a[k],b[k],rtol=1e-6,atol=1e-7)
 for k in ('rng','scaler','physical','step','target_steps','initial_state_sha256','u_seen'):train.assert_tree(a[k],b[k],rtol=0,atol=0)
def compare_records(a,b):
 for k in a.keys()-{'seconds','updated_utc','extra_strong_forward_pair_visits'}:
  if isinstance(a[k],float):
   if abs(a[k]-b[k])>1e-6:raise RuntimeError('Original numerical record gate failed '+k)
  elif a[k]!=b[k]:raise RuntimeError('Original logical record differs '+k)
def benchmark(crop,variant,steps=24):
 rid=crop+'_010_freematch_s20260825';cfg,contract,prov,_=train.check(rid,True)
 folder=ROOT/'audit/speed_v4/benchmarks'/crop/variant
 if folder.exists():raise RuntimeError('Existing engineering attempt preserved, no implicit retry')
 folder.mkdir(parents=True)
 started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 runner=train.Runner(cfg,contract,prov) if variant=='reference' else SpeedRunner(cfg,contract,prov,checkpointing=variant=='retained_checkpoint',decoded_capacity=256)
 records=[]
 for _ in range(steps):
  rec=runner.train_step();records.append(rec)
  if runner.step in (1,2,steps):
   payload=runner.snapshot()
   payload['engineering_execution']=dict(variant=variant,actual_activation_checkpointing=getattr(runner,'actual_activation_checkpointing',True),formal_steps_added=0,science_profile_sha256=train.digest(ROOT/'configs/execution_profile.json'),actual_engineering_source=read(ROOT/'audit/speed_v4/cpu_preflight.json')['source'])
   atomic_torch(folder/('step%d.pt'%runner.step),payload)
 atomic_json(folder/'records.json',records)
 torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
 if variant!='reference':
  ref=folder.parent/'reference'
  compare(torch.load(ref/'step2.pt',map_location='cpu',weights_only=False),torch.load(folder/'step2.pt',map_location='cpu',weights_only=False))
  compare(torch.load(ref/('step%d.pt'%steps),map_location='cpu',weights_only=False),torch.load(folder/('step%d.pt'%steps),map_location='cpu',weights_only=False))
  prior=read(ref/'records.json')
  for a,b in zip(prior,records):compare_records(a,b)
  first=torch.load(folder/'step1.pt',map_location='cpu',weights_only=False)
  del runner;gc.collect();torch.cuda.empty_cache()
  runner=SpeedRunner(cfg,contract,prov,checkpointing=variant=='retained_checkpoint',decoded_capacity=256)
  train.reconstruct_first_cache(runner);runner.load(first);replayed=runner.train_step()
  compare_records(records[1],replayed)
  compare(torch.load(folder/'step2.pt',map_location='cpu',weights_only=False),runner.snapshot())
 result=dict(passed=True,crop=crop,variant=variant,engineering_steps=steps,formal_steps_added=0,no_Validation_Test=True,median_seconds=statistics.median(x['seconds'] for x in records[6:]),mean_seconds=statistics.mean(x['seconds'] for x in records[6:]),peak_allocated_bytes=peak,full_reference_student_EMA_optimizer_scaler_rng_SAT_records_verified=variant!='reference',cold1_to2_replay_passed=variant!='reference',seconds=time.monotonic()-started,created_utc=now())
 atomic_json(folder/'passed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 benchmark(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 24)
