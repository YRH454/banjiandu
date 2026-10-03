"""Single authorized whole-boundary execution acceleration; GPU tests before deployment."""
import gc,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(Path(__file__).parent))
os.environ['CUDA_VISIBLE_DEVICES']=''
import train,dispatcher as original
import boundary
from common import now,atomic_json,read,digest,exclusive
HERE=ROOT/'audit/speed_v4';EXT=ROOT/'orchestration/speed_v4'
def state(label,**kw):atomic_json(HERE/'job_state.json',dict(state=label,identity=original.identity(os.getpid()),updated_utc=now(),**kw))
def command(args,gpu=False,label='CPU'):
 so=ROOT/'logs'/('speed_engineering_'+label+'_'+str(time.time_ns())+'.stdout.log');se=so.with_name(so.name.replace('.stdout.','.stderr.'))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=train.GPU if gpu else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false')
 with so.open('xb') as o,se.open('xb') as e:p=subprocess.Popen([sys.executable,'-B','-u']+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=o,stderr=e)
 state('engineering_'+label,child=original.identity(p.pid),stdout=str(so),stderr=str(se));code=p.wait()
 return code,so,se
def memory_bounds(runs):
 from albef_ssl.model import get_tokenizer
 tok=get_tokenizer(str(ROOT/'assets/tokenizer'));bounds={}
 for rid in runs:
  cfg=train.config_for(rid);m,assets,l,u,val=train.load_contract(cfg['dataset'],cfg['budget'])
  train_images={x['image_id'] for x in l+u};val_images={x['image_id'] for x in val}
  strings={x['text'] for x in u}
  for x in l+val:strings.update((x['positive_text'],x['negative_text']))
  lengths=tok(list(strings),truncation=False,padding=False,return_attention_mask=False)['input_ids']
  if any(len(x)>m['token_guard'] for x in lengths):raise RuntimeError('Complete caption over immutable guard')
  prefix=(2*len(train_images)+len(val_images))*577*768*2
  text=sum(len(x)*768*2 for x in lengths)
  bounds[rid]=prefix+text+5*1024**3+256*384*384*3
 return bounds
def main():
 capture=None;retired=False
 with exclusive(ROOT/'locks/speed_engineering_single.lock'):
  try:
   source={str(p.relative_to(ROOT)):digest(p) for p in sorted(EXT.glob('*.py'))}
   code,so,se=command(['-m','unittest','discover','-s','orchestration/speed_v4','-p','cpu_tests.py','-v'],label='CPU_tests')
   if code:raise RuntimeError('Independent CPU admission failed')
   atomic_json(HERE/'cpu_preflight.json',dict(passed=True,source=source,stdout=str(so),stderr=str(se),original_source_config_unchanged=True,created_utc=now()))
   state('waiting_for_full_complete_group_boundary_no_GPU')
   capture=boundary.capture()
   if not capture:
    state('safe_boundary_not_captured_original_continues');return
   atomic_json(HERE/'completed_original_proofs.json',{rid:train.completed(rid) for rid in capture['completed']})
   results={};failures=[]
   with exclusive(ROOT/'locks'/(train.GPU+'.lock')):
    if original.compute():raise RuntimeError('No engineering beside scientific worker')
    for crop in train.CROPS:
     results[crop]={}
     for variant in ('reference','retained_checkpoint','retained_no_checkpoint'):
      code,so,se=command(['orchestration/speed_v4/bench.py',crop,variant,'24'],gpu=True,label=crop+'_'+variant)
      if code:
       failures.append(dict(crop=crop,variant=variant,stdout=str(so),stderr=str(se),preserved=True))
       if variant=='reference':raise RuntimeError('Original reference engineering failed; preserve/revert')
      else:results[crop][variant]=read(HERE/'benchmarks'/crop/variant/'passed.json')
   choices=[v for v in ('retained_no_checkpoint','retained_checkpoint') if all(v in results[c] for c in train.CROPS)]
   variant=min(choices,key=lambda v:sum(results[c][v]['median_seconds'] for c in train.CROPS)) if choices else 'reference'
   original_time=sum(results[c]['reference']['median_seconds'] for c in train.CROPS)
   accelerated_time=sum(results[c][variant]['median_seconds'] for c in train.CROPS)
   if accelerated_time>=original_time*.97:variant='reference';accelerated_time=original_time
   mem=int(Path('/sys/fs/cgroup/memory.max').read_text());ram=memory_bounds(capture['untouched'])
   gpu_lines=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()
   gpu_total=int(next(x for x in gpu_lines if train.GPU in x).split(',')[-1])*1024**2
   peaks={rid:results[train.config_for(rid)['dataset']][variant]['peak_allocated_bytes']+512*1024**2 for rid in capture['untouched']}
   slots=2 if sum(sorted(set(peaks.values()))[-2:])<gpu_total*.9 else 1
   # RAM-safe pairing, not unbounded GPU launch. Actual two-process throughput tested first.
   import supervisor
   reg=dict(version='freematch_source_preserving_GPU_speed_v4',source=source,original_profile_sha256=boundary.EXPECTED_PROFILE,eligible_runs=capture['untouched'],target_steps=2600,variant=variant,max_workers=slots,RAM_upper_bytes=ram,memory_limit_bytes=mem,GPU_peak_bytes=peaks,GPU_total_bytes=gpu_total,original_science_configs_unchanged=True,actual_execution_overrides=dict(retained_physical16_strong_graphs=variant!='reference',activation_checkpointing=variant in ('reference','retained_checkpoint'),bounded_exact_decoded_image_LRU=0 if variant=='reference' else 256,unique_wave_max_workers=slots),no_scientific_worker_migration=True,registered_utc=now())
   # Concurrency engineering uses only isolated independent fresh-runner children.
   wave=supervisor.planned_wave(capture['untouched'],reg)
   concurrent=None
   if slots==2 and len(wave)==2:
    with exclusive(ROOT/'locks'/(train.GPU+'.lock')):
     child_jobs=[]
     for rid in wave:
      crop=train.config_for(rid)['dataset'];label='parallel_'+crop
      folder=HERE/'benchmarks'/crop/('parallel_'+variant)
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=train.GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false')
      so=ROOT/'logs'/('speed_parallel_'+crop+'_'+str(time.time_ns())+'.stdout.log');se=so.with_name(so.name.replace('.stdout.','.stderr.'))
      with so.open('xb') as o,se.open('xb') as e:p=subprocess.Popen([sys.executable,'-B','-u','orchestration/speed_v4/parallel_bench.py',crop,variant],cwd=ROOT,env=env,stdout=o,stderr=e,stdin=subprocess.DEVNULL)
      child_jobs.append((p,crop,so,se))
     state('authorized_two_engineering_workers_not_formal',children=[original.identity(p.pid) for p,c,so,se in child_jobs])
     for p,c,so,se in child_jobs:p.wait()
     if all(p.returncode==0 for p,c,so,se in child_jobs):
      concurrent={c:read(HERE/'benchmarks'/c/('parallel_'+variant)/'passed.json') for p,c,so,se in child_jobs}
      sum_serial=sum(results[c][variant]['median_seconds'] for c in concurrent)
      slow=max(v['median_seconds'] for v in concurrent.values())
      if slow>=sum_serial*.95:reg['max_workers']=1
     else:
      reg['max_workers']=1;failures.extend(dict(crop=c,variant='parallel_'+variant,stdout=str(so),stderr=str(se),preserved=True) for p,c,so,se in child_jobs if p.returncode)
   else:reg['max_workers']=1
   if variant=='reference' and reg['max_workers']==1:raise RuntimeError('Neither exact numerical optimization nor real parallel throughput admitted; keep original serial')
   reg['actual_execution_overrides']['unique_wave_max_workers']=reg['max_workers']
   atomic_json(EXT/'registration.json',reg)
   admission=dict(passed=True,registration_sha256=digest(EXT/'registration.json'),original_full_state_and_records_gates_not_relaxed=True,all_four_crop_real_GPU_benchmarks=results,all_failed_attempts_preserved=failures,selected=variant,max_workers=reg['max_workers'],two_process_real_GPU_throughput=concurrent,median_single_worker_speed_ratio=original_time/accelerated_time,created_utc=now())
   atomic_json(HERE/'admission.json',admission)
   code,so,se=command(['-c','import sys;sys.path.insert(0,"orchestration/speed_v4");import runtime;runtime.registration();import supervisor;assert supervisor.original_dispatcher.__file__.endswith("/code/dispatcher.py");print("registered CPU runtime prelaunch passes")'],label='runtime_prelaunch')
   if code:raise RuntimeError('Fresh supervisor CPU prelaunch failed')
   # Recheck all preserved original output bytes before retiring only its CPU controller.
   for rid,hashes in capture['completed_hashes'].items():
    for n,h in hashes.items():
     if digest(ROOT/'outputs'/rid/n)!=h:raise RuntimeError('Completed original bytes changed')
   boundary.retire(capture);retired=True
   stamp=str(time.time_ns());so=ROOT/'logs'/('speed_supervisor_'+stamp+'.stdout.log');se=so.with_name(so.name.replace('.stdout.','.stderr.'))
   with so.open('xb') as o,se.open('xb') as e:p=subprocess.Popen([sys.executable,'-B','-u','orchestration/speed_v4/supervisor.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True)
   atomic_json(HERE/'production_launch.json',dict(supervisor=original.identity(p.pid),registration_sha256=digest(EXT/'registration.json'),stdout=str(so),stderr=str(se),only_old_CPU_controller_retired_at_verified_boundary=True,scientific_workers_not_stopped=True,created_utc=now()))
   state('new_admitted_speed_supervisor_launched',supervisor=original.identity(p.pid),registration_sha256=digest(EXT/'registration.json'),variant=variant,max_workers=reg['max_workers'])
  except BaseException:
   error=traceback.format_exc();atomic_json(HERE/'job_failure.json',dict(error=error,original_scientific_worker_not_stopped=True,original_controller_retired=retired,created_utc=now()))
   if capture and not retired and not original.compute():boundary.resume(capture);state('engineering_not_admitted_original_serial_resumed',error=error)
   else:state('diagnosis_required_preserve_all',error=error)
   raise
if __name__=='__main__':main()
