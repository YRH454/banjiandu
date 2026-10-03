"""CPU-only admission checks; no scientific GPU or formal output writes."""
import ast,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class Tests(unittest.TestCase):
 def test_sources_parse(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
 def test_original_frozen(self):
  p=json.loads((ROOT/'configs/execution_profile.json').read_text())
  for n,h in p['source'].items():self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h)
 def test_original_config_frozen(self):
  p=json.loads((ROOT/'configs/execution_profile.json').read_text())
  for n,h in p['config_hashes'].items():self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h)
 def test_successful_update_unchanged(self):
  s=(Path(__file__).parent/'candidates.py').read_text()
  for statement in ('self.algorithm.commit(proposed)','self.step = step','self.scaler.step(self.opt)','self.opt.step()','loss = raw_loss * self.cfg["training"]["lambda_u"]'):
   self.assertIn(statement,s)
 def test_global_SAF_unchanged(self):
  s=(Path(__file__).parent/'candidates.py').read_text()
  self.assertIn('saf_value_and_gradient(strong, mask, proposed)',s)
  self.assertIn('self.cfg["method_settings"]["ent_loss_ratio"]',s)
 def test_no_extra_strong_cost_claim(self):
  s=(Path(__file__).parent/'candidates.py').read_text()
  self.assertIn('"extra_strong_forward_pair_visits":0',s)
  self.assertNotIn('"extra_strong_forward_pair_visits":step*32',s)
 def test_no_formal_api_in_candidates(self):
  s=(Path(__file__).parent/'candidates.py').read_text()
  self.assertNotIn('base.run(',s);self.assertNotIn('train.gate(',s)
 def test_retry_graph_cleanup(self):
  s=(Path(__file__).parent/'candidates.py').read_text()
  self.assertEqual(s.count('strong_graphs = None'),2)
 def test_original_microbatch_and_divisor(self):
  s=(Path(__file__).parent/'candidates.py').read_text()
  self.assertIn('chunks(us, self.physical)',s);self.assertIn('raw_loss = values.sum() / 32',s);self.assertIn('self.physical //= 2',s)

class WaveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import os,sys
  os.environ['CUDA_VISIBLE_DEVICES']=''
  sys.path.insert(0,str(Path(__file__).parent))
  import supervisor
  cls.plan=staticmethod(supervisor.planned_wave)
 def rules(self):return dict(max_workers=2,RAM_upper_bytes=dict(a=40,b=30,c=10),memory_limit_bytes=60,GPU_peak_bytes=dict(a=2,b=2,c=2),GPU_total_bytes=10)
 def test_memory_safe_pair(self):self.assertEqual(self.plan(['a','b','c'],self.rules()),['a','c'])
 def test_single_capacity(self):
  r=self.rules();r['max_workers']=1;self.assertEqual(self.plan(['a','c'],r),['a'])
 def test_GPU_capacity_refuses(self):
  r=self.rules();r['GPU_total_bytes']=3;self.assertEqual(self.plan(['a','c'],r),['a'])
 def test_empty(self):self.assertEqual(self.plan([],self.rules()),[])
 def test_duplicate_refused(self):
  with self.assertRaises(RuntimeError):self.plan(['a','a'],self.rules())
class LockTests(unittest.TestCase):
 def probe(self,path,mode):
  import subprocess,sys
  code="import fcntl,sys;f=open(sys.argv[1],'r+b',buffering=0)\ntry:fcntl.flock(f, (fcntl.LOCK_SH if sys.argv[2]=='SH' else fcntl.LOCK_EX)|fcntl.LOCK_NB)\nexcept BlockingIOError:sys.exit(4)\nsys.exit(0)"
  return subprocess.run([sys.executable,'-B','-c',code,str(path),mode],timeout=10).returncode
 def case(self,parent,child,expected,release=False,different=False):
  import os,tempfile
  if os.name!='posix':self.skipTest('Real POSIX proof runs on authorized Linux')
  import fcntl
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lock';p.write_bytes(b'\0');q=Path(d)/'other';q.write_bytes(b'\0')
   with p.open('r+b',buffering=0) as f:
    fcntl.flock(f,fcntl.LOCK_SH if parent=='SH' else fcntl.LOCK_EX)
    if release:fcntl.flock(f,fcntl.LOCK_UN)
    self.assertEqual(self.probe(q if different else p,child),expected)
 def test_two_shared_slots(self):self.case('SH','SH',0)
 def test_shared_blocks_old_exclusive(self):self.case('SH','EX',4)
 def test_old_exclusive_blocks_shared(self):self.case('EX','SH',4)
 def test_unique_run_exclusive(self):self.case('EX','EX',4)
 def test_release_allows_next(self):self.case('SH','EX',0,release=True)
 def test_run_paths_independent(self):self.case('EX','EX',0,different=True)
 def test_exclusive_engineering_after_release(self):self.case('EX','SH',0,release=True)

class NativeIdentityTests(unittest.TestCase):
 def test_self_pidfd_exact(self):
  import os,sys
  if os.name!='posix':self.skipTest('Actual Linux native process handle proof')
  sys.path.insert(0,str(Path(__file__).parent));import boundary
  fd=boundary.pidfd_open(os.getpid())
  try:boundary.pidfd_send_signal(fd,0)
  finally:os.close(fd)
 def test_stale_pid_rejected(self):
  import os,sys
  if os.name!='posix':self.skipTest('Actual Linux native process handle proof')
  sys.path.insert(0,str(Path(__file__).parent));import boundary
  with self.assertRaises(OSError):boundary.pidfd_open(2147483647)

class FullInputAndProvenanceTests(unittest.TestCase):
 def test_current_full_public_images_guard(self):
  s=(Path(__file__).parent/'supervisor.py').read_text()
  self.assertIn("bootstrap.crop_ready",s)
 def test_engineering_reference_truthful_version(self):
  s=(Path(__file__).parent/'runtime.py').read_text()
  self.assertIn("original_science_checkpointed_two_pass_reference",s)
  self.assertIn("if k!='execution_extension'",s)

class ReleaseAndFallbackTests(unittest.TestCase):
 def test_bounded_native_GPU_release(self):
  s=(Path(__file__).parent/'boundary.py').read_text();self.assertIn('release_deadline=time.monotonic()+30',s)
 def test_reference_parallel_preserves_original(self):
  s=(Path(__file__).parent/'runtime.py').read_text();self.assertIn("Parent=OriginalRunner if r['variant']=='reference'",s)
 def test_actual_engineering_flag_from_runner(self):
  s=(Path(__file__).parent/'bench.py').read_text();self.assertIn("getattr(runner,'actual_activation_checkpointing',True)",s)

if __name__=='__main__':unittest.main(verbosity=2)
