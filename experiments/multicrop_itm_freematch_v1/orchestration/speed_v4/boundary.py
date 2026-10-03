"""Exact whole-group capture: never suspend/stop a scientific worker."""
import gc,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
os.environ['CUDA_VISIBLE_DEVICES']=''
import train,dispatcher
from common import now,atomic_json,exclusive,digest,read
HERE=ROOT/'audit/speed_v4'
EXPECTED_PROFILE='7f142af99114c8a968744708624ac483e285bf353a8f5fa5822366546c28aefb'
import ctypes,platform
def pidfd_open(pid):
 if platform.machine()!='x86_64':raise RuntimeError('Only actual verified Linux x86_64 syscall mapping')
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 fd=libc.syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))
 if fd<0:raise OSError(ctypes.get_errno(),'native pidfd_open failed')
 return int(fd)
def pidfd_send_signal(fd,signum):
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 result=libc.syscall(424,ctypes.c_int(fd),ctypes.c_int(signum),ctypes.c_void_p(),ctypes.c_uint(0))
 if result<0:raise OSError(ctypes.get_errno(),'native pidfd_send_signal failed')

def inspect_lock(path):
 import fcntl
 if not path.exists():return False
 with path.open('rb',buffering=0) as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN);return False
def same(a,b):
 return bool(a and b and a['pid']==b['pid'] and a['start_ticks']==b['start_ticks'] and a['command']==b['command'])
def capture(timeout=5400):
 train.profile(True)
 s=read(ROOT/'outputs/pipeline_state.json')
 init_deadline=time.monotonic()+180
 while s.get('phase')!='run' and s['version']=='freematch_server3_serial20_v1' and time.monotonic()<init_deadline:
  atomic_json(HERE/'boundary_state.json',dict(state='waiting_original_check_gate_initialization_no_GPU',updated_utc=now()))
  time.sleep(.25);s=read(ROOT/'outputs/pipeline_state.json')
 expected=s['identity'];target=s['active_run']
 if s['phase']!='run' or target not in train.RUNS or s['version']!='freematch_server3_serial20_v1':raise RuntimeError('Only currently observed original serial whole group; not an unrelated controller')
 if digest(ROOT/'configs/execution_profile.json')!=EXPECTED_PROFILE:raise RuntimeError('Original source version differs')
 deadline=time.monotonic()+timeout;fd=pidfd_open(expected['pid'])
 try:
  while time.monotonic()<deadline:
   if not same(dispatcher.identity(expected['pid']),expected):raise RuntimeError('Old controller exact identity gone')
   current=read(ROOT/'outputs/pipeline_state.json')
   status_file=ROOT/'outputs'/target/'status.json'
   status=read(status_file) if status_file.exists() else {}
   if current.get('active_run') not in (target,None):return None
   atomic_json(HERE/'boundary_state.json',dict(state='waiting_complete_group_boundary',coordinator=dispatcher.identity(os.getpid()),old_controller=expected,active_run=target,step=status.get('step'),updated_utc=now()))
   child=current.get('child');alive=dispatcher.identity(child['pid']) if child else None
   zombie=False
   if alive:
    st=(Path('/proc')/str(alive['pid'])/'stat').read_text();zombie=st[st.rfind(')')+2:].split()[0]=='Z'
   if status.get('state')=='completed' and status.get('step')==2600 and (not alive or zombie):
    pidfd_send_signal(fd,signal.SIGSTOP)
    try:
     # CUDA/NVML and flock cleanup can lag /proc zombie transition; wait, never kill.
     release_deadline=time.monotonic()+30
     while True:
      apps=dispatcher.compute()
      occupied=inspect_lock(ROOT/'locks'/(train.GPU+'.lock')) or inspect_lock(ROOT/'locks'/(target+'.lock'))
      if not apps and not occupied:break
      allowed={child['pid']} if child else set()
      if any(int(line.split(',')[1].strip()) not in allowed for line in apps):raise RuntimeError('Unexpected different GPU worker; resume old without interference')
      if time.monotonic()>release_deadline:raise RuntimeError('Actual CUDA/lock release not captured; preserve original continuation')
      atomic_json(HERE/'boundary_state.json',dict(state='verified_worker_exited_waiting_native_GPU_lock_release',old_controller=expected,completed_run=target,compute=apps,updated_utc=now()))
      time.sleep(.1)
     current=read(ROOT/'outputs/pipeline_state.json')
     if current.get('active_run') not in (target,None):raise RuntimeError('Missed boundary, preserve original continuation')
     children=[]
     for p in Path('/proc').iterdir():
      if p.name.isdigit():
       i=dispatcher.identity(int(p.name))
       if i and i['ppid']==expected['pid']:
        st=(p/'stat').read_text();state=st[st.rfind(')')+2:].split()[0]
        if state!='Z':children.append(i)
     if children:raise RuntimeError('Another child already launched, do not retire')
     if inspect_lock(ROOT/'locks'/(train.GPU+'.lock')) or inspect_lock(ROOT/'locks'/(target+'.lock')):raise RuntimeError('Actual scientific locks still held')
     completed=[rid for rid in train.RUNS if (ROOT/'outputs'/rid/'result.json').exists()]
     proofs={rid:train.completed(rid) for rid in completed}
     untouched=[rid for rid in train.RUNS if rid not in completed]
     for rid in untouched:
      o=ROOT/'outputs'/rid
      if any((o/n).exists() for n in ('status.json','last.pt','best.pt','result.json')) or list(o.glob('train_*.jsonl')):raise RuntimeError('Future group already formally started')
     receipt=dict(state='verified_complete_group_boundary_controller_suspended',old_controller=expected,captured_run=target,completed=completed,untouched=untouched,completed_hashes={rid:proof['all_output_hashes'] for rid,proof in proofs.items()},original_profile_sha256=EXPECTED_PROFILE,source_unchanged=True,scientific_worker_not_terminated=True,created_utc=now())
     atomic_json(HERE/'boundary_captured.json',receipt)
     return receipt
    except BaseException:
     pidfd_send_signal(fd,signal.SIGCONT);raise
   time.sleep(.1)
  return None
 finally:os.close(fd)
def resume(capture):
 expected=capture['old_controller']
 if same(dispatcher.identity(expected['pid']),expected):
  fd=pidfd_open(expected['pid'])
  try:pidfd_send_signal(fd,signal.SIGCONT)
  finally:os.close(fd)
 atomic_json(HERE/'boundary_state.json',dict(state='original_serial_queue_resumed_no_worker_restart',updated_utc=now()))
def retire(capture):
 expected=capture['old_controller']
 if not same(dispatcher.identity(expected['pid']),expected) or dispatcher.compute():raise RuntimeError('Retirement identity/GPU guard differs')
 fd=pidfd_open(expected['pid'])
 try:
  pidfd_send_signal(fd,signal.SIGTERM);pidfd_send_signal(fd,signal.SIGCONT)
  deadline=time.monotonic()+15
  while same(dispatcher.identity(expected['pid']),expected):
   p=Path('/proc')/str(expected['pid'])/'stat'
   if p.exists():
    st=p.read_text();z=st[st.rfind(')')+2:].split()[0]=='Z'
    if z:break
   if time.monotonic()>deadline:raise RuntimeError('Retired controller still present; no new dispatcher')
   time.sleep(.1)
 finally:os.close(fd)
 if inspect_lock(ROOT/'locks/dispatcher_single.lock'):raise RuntimeError('Old dispatcher lock remains')
 atomic_json(HERE/'retired.json',dict(old_controller=expected,exact_whole_group_boundary=True,no_scientific_worker_terminated=True,created_utc=now()))
if __name__=='__main__':
 with exclusive(ROOT/'locks/speed_engineering_single.lock'):
  receipt=capture();print(json.dumps(receipt),flush=True)
