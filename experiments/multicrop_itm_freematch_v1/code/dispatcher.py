"""Unique serial fifth20 dispatcher. Full failures stop; never resume formal work."""
import gc,json,os,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,train,bootstrap
from common import ROOT,read,digest,now,atomic_json,exclusive

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
        return dict(pid=pid,ppid=int(f[1]),start_ticks=f[19],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,PermissionError,ProcessLookupError):return None
def compute():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return [line for line in r.stdout.splitlines() if train.GPU in line]
def save(s,label,**kw):s.update(state=label,updated_utc=now(),**kw);atomic_json(ROOT/'outputs/pipeline_state.json',s)
def main():
    profile=train.profile(True);torch.set_num_threads(2)
    state=dict(version='freematch_server3_serial20_v1',identity=identity(os.getpid()),gpu_uuid=train.GPU,target_steps=2600,execution_profile_sha256=digest(ROOT/'configs/execution_profile.json'),completed=[],automatic_formal_recovery_authorized=False)
    with exclusive(ROOT/'locks/dispatcher_single.lock'):
        try:
            for rid in train.RUNS:
                if (ROOT/'outputs'/rid/'result.json').exists():
                    proof=train.completed(rid);atomic_json(ROOT/'audit/completed'/(rid+'.json'),proof);state['completed'].append(rid);continue
                if any((ROOT/'outputs'/rid/n).exists() for n in ('status.json','last.pt','best.pt')) or list((ROOT/'outputs'/rid).glob('train_*.jsonl')):raise RuntimeError('Existing formal evidence: separate user recovery authorization required')
                crop=train.config_for(rid)['dataset']
                while not (ROOT/'audit/verified_crops'/(crop+'.json')).exists():
                    save(state,'waiting_for_verified_crop_images',active_run=rid,phase='CPU_waiting',child=None)
                    if (ROOT/'audit/crop_images_delivery'/(crop+'.json')).exists():bootstrap.crop_ready(crop)
                    else:time.sleep(15)
                if not bootstrap.crop_ready(crop):raise RuntimeError('Current full crop input receipt absent')
                if compute():raise RuntimeError('Assigned RTX4090 occupied; no implicit termination')
                for phase in ('check','gate','run'):
                    stamp=str(time.time_ns());so=ROOT/'logs'/(rid+'_'+phase+'_'+stamp+'.stdout.log');se=ROOT/'logs'/(rid+'_'+phase+'_'+stamp+'.stderr.log')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if phase=='check' else train.GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false')
                    cmd=[sys.executable,'-B','-u','code/train.py','--run-id',rid,'--'+phase]
                    if phase!='check':cmd+=['--gpu-uuid',train.GPU]
                    with so.open('xb') as out,se.open('xb') as err:
                        child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env)
                        save(state,'engineering_'+phase if phase!='run' else 'formal_training',active_run=rid,phase=phase,child=identity(child.pid),child_command=cmd,stdout=str(so),stderr=str(se))
                        while child.poll() is None:
                            sf=ROOT/'outputs'/rid/'status.json';save(state,state['state'],status=read(sf) if sf.exists() else None);time.sleep(5)
                    save(state,state['state'],child_exit_code=child.returncode)
                    if child.returncode:raise RuntimeError('Fifth '+phase+' failed; preserve evidence, no automatic retry')
                proof=train.completed(rid);atomic_json(ROOT/'audit/completed'/(rid+'.json'),proof);state['completed'].append(rid)
                save(state,'run_full_completed',last_complete_run=rid,active_run=None,phase=None,child=None);gc.collect()
            proofs={rid:read(ROOT/'audit/completed'/(rid+'.json')) for rid in train.RUNS}
            atomic_json(ROOT/'reports/fifth_summary.json',dict(configurations=20,all_fully_verified=True,target_steps=2600,results=proofs,source=profile['source'],execution_profile_sha256=digest(ROOT/'configs/execution_profile.json'),extra_strong_forward_pairs_per_run=2600*32,limits='Single seed; random weak false negatives; Validation threshold/model selection; binary image-caption adaptation, not unmodified image-classification reproduction; unequal successful-step/compute budgets versus MT/FM/SoftMatch/SimMatch; global SAF second strong forward cost measured in wall/step time; actual LinuxRTX4090 environment; engineering excluded; no Test',completed_utc=now()))
            save(state,'all20_full_completed_summary_verified',active_run=None,child=None)
        except BaseException:save(state,'failed_preserve_all_no_automatic_recovery',error=traceback.format_exc());raise
if __name__=='__main__':main()
