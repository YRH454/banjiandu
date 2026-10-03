"""Independent FreeMatch binary pair-ITM adapter; frozen model/data/evaluation reused.
Not an exact image-classification paper reproduction. 2600 successful steps.
"""
import argparse,copy,gc,json,math,os,random,socket,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
import base_train as base
from common import ROOT,read,digest,canonical,now,atomic_json,atomic_torch,exclusive,code_hashes
from data_backend import PrefixCache,fixed_seed,flatten_l,chunks,load_contract
from losses import weighted_bce
from freematch_algorithm import FreeMatchState,saf_value_and_gradient
from initialization import bind_zero
GPU='GPU-00000000-0000-0000-0000-000000000000'
NAMED='b42eca74de637ed54ecaeea5268645a42dc0a9931ee45a2917591d4a1848704c'
RAW='3dd177e33a5a7f2f3ac7e43dd8a28236bdc4d20118af3df1381a4752a98ace74'
CROPS=('apple','cassava','rice','banana');BUDGETS=('001','005','010','020','030')
RUNS=[c+'_'+b+'_freematch_s20260825' for c in CROPS for b in BUDGETS]
def profile(require_CPU=False):
    p=read(ROOT/'configs/execution_profile.json')
    if socket.gethostname()!='EXAMPLE-HOST' or str(ROOT)!='/path/to/freematch-experiment':raise RuntimeError('Actual registered fifth host/root differs')
    if p['run_ids']!=RUNS or p['target_steps']!=2600 or p['gpu_uuid']!=GPU:raise RuntimeError('Independent fifth scope differs')
    if p['source']!=code_hashes():raise RuntimeError('Registered fifth source changed')
    for n,h in p['tools_source'].items():
        if digest(ROOT/n)!=h:raise RuntimeError('Registered engineering tool changed '+n)
    for n,h in p['input_contract_hashes'].items():
        if digest(ROOT/n)!=h:raise RuntimeError('Frozen full public contract changed '+n)
    if p['environment']!=base.versions():raise RuntimeError('Actual fifth execution environment differs')
    for n,h in p['config_hashes'].items():
        if digest(ROOT/n)!=h:raise RuntimeError('Registered scientific config changed '+n)
    if require_CPU:
        cpu=read(ROOT/'audit/cpu_preflight.json')
        if not cpu['passed'] or cpu['execution_profile_sha256']!=digest(ROOT/'configs/execution_profile.json'):raise RuntimeError('Actual fifth CPU proof absent')
    return p
def config_for(rid):
    if rid not in RUNS:raise RuntimeError('Only separately authorized fifth20 configs')
    cfg=read(ROOT/'configs/runs'/(rid+'.json'));protocol=read(ROOT/'configs/protocol.json')
    if cfg['run_id']!=rid or cfg['method']!='freematch' or cfg['target_steps']!=2600 or cfg['seed']!=20260825 or not cfg['launch_enabled'] or protocol['targets']!={'freematch':2600} or digest(ROOT/'configs/protocol.json')!=cfg['protocol_sha256']:raise RuntimeError('Fifth scientific scope/steps changed')
    return cfg
OriginalRunner=base.Runner
class Runner(OriginalRunner):
    def __init__(self,cfg,contract,prov):
        super().__init__(cfg,contract,prov)
        if self.initial_hash!=NAMED:raise RuntimeError('Registered exact common34 initial values differ')
        self.algorithm=FreeMatchState(cfg['method_settings']['ema_p'],self.device)
    def snapshot(self):
        if self.algorithm.updates!=self.step:raise RuntimeError('SAT state must match successful step')
        value=super().snapshot()
        value.update(algorithm_checkpoint_version='freematch_full_v1',algorithm_state=self.algorithm.state_dict())
        return value
    def load(self,payload):
        if payload.get('algorithm_checkpoint_version')!='freematch_full_v1' or 'algorithm_state' not in payload or payload['algorithm_state']['updates']!=payload['step']:raise RuntimeError('Full SAT/SAF checkpoint required')
        super().load(payload);self.algorithm.load(payload['algorithm_state'])
    def train_step(self):
        step = self.step + 1
        seed = self.cfg["seed"]
        chosen_l = random.Random(fixed_seed(seed, step, "pair-sampler")).sample(self.l, 16)
        chosen_u = random.Random(fixed_seed(seed, step, "u-sampler")).sample(self.u, 32)
        lp = flatten_l(chosen_l)
        targets_l = torch.tensor([p["label"] for p in lp], device=self.device)
        timer = time.monotonic()
        attempts = 0
        fp32 = False
        while True:
            attempts += 1
            if attempts > 12:
                raise RuntimeError("Repeated same-step failures; stopping without counting update")
            self.opt.zero_grad(set_to_none=True)
            self.model.fusion_chunk_size = self.ema.fusion_chunk_size = self.physical
            lb = uw = us = weak = z = loss = raw_loss = inputs = target_u = mask = strong = fairness = fairness_grad = proposed = thresholds = None
            try:
                lb = self.cache.batch(lp, step, "weak", self.physical)
                uw = self.cache.batch(chosen_u, step, "weak", self.physical)
                us = self.cache.batch(chosen_u, step, "strong", self.physical)
                weak = self.predict(self.model, uw, amp=not fp32)
                target_u, mask, proposed, thresholds = self.algorithm.propose(weak)
                strong = self.predict(self.model, us, amp=not fp32)
                fairness, fairness_grad = saf_value_and_gradient(strong, mask, proposed)
                ramp = 1.
                warmup = self.cfg["training"]["warmup_steps"]
                factor = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step-warmup) / (self.cfg["target_steps"]-warmup)))
                for g in self.opt.param_groups:
                    g["lr"] = g["base_lr"] * factor
                sup_total, unsup_total = 0., 0.
                for i, x in enumerate(chunks(lb, self.physical)):
                    begin = i * self.physical
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        z, _ = self.model.forward_pairs(*inputs)
                        loss = F.binary_cross_entropy_with_logits(z.float(), targets_l[begin:begin+len(z)], reduction="sum") / 32
                    self.scaler.scale(loss).backward() if not fp32 else loss.backward()
                    sup_total += float(loss.detach())
                for i, x in enumerate(chunks(us, self.physical)):
                    begin = i * self.physical
                    inputs = tuple(v.float() if fp32 and v.is_floating_point() else v for v in x)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        z, _ = self.model.forward_pairs(*inputs)
                        tar, weights = target_u[begin:begin+len(z)], mask[begin:begin+len(z)]
                        torch.testing.assert_close(z.float(), strong[begin:begin+len(z)], rtol=0, atol=0)
                        values = weighted_bce(z, tar, weights)
                        raw_loss = values.sum() / 32
                        loss = raw_loss * self.cfg["training"]["lambda_u"]
                        loss = loss + self.cfg["method_settings"]["ent_loss_ratio"] * (z.float()*fairness_grad[begin:begin+len(z)]).sum()
                    self.scaler.scale(loss).backward() if not fp32 else loss.backward()
                    unsup_total += float(raw_loss.detach())
                if not fp32:
                    self.scaler.unscale_(self.opt)
                finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.params)
                if not finite:
                    if fp32:
                        raise FloatingPointError("Nonfinite FP32 gradient")
                    scale = self.scaler.get_scale()
                    self.scaler.update(new_scale=max(1., scale / 2))
                    fp32 = scale <= 16
                    del lb, uw, us, weak
                    continue
                norm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["training"]["max_grad_norm"])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite clipping norm")
                if fp32:
                    self.opt.step()
                else:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                with torch.no_grad():
                    teacher_params = dict(self.ema.named_parameters())
                    m = self.cfg["training"]["ema_decay"]
                    for n, p in self.model.named_parameters():
                        if p.requires_grad:
                            teacher_params[n].mul_(m).add_(p, alpha=1-m)
                self.algorithm.commit(proposed)
                self.step = step
                self.u_seen.update(p["pair_id"] for p in chosen_u)
                seconds = time.monotonic()-timer
                self.elapsed_seconds += seconds
                return {"step": step, "sup_loss": sup_total, "unsup_loss": unsup_total,
                        "saf_loss":float(fairness),"threshold_global":float(self.algorithm.time_p),
                        "threshold_negative":float(thresholds[0]),"threshold_positive":float(thresholds[1]),
                        "algorithm_successful_updates":self.algorithm.updates,
                        "extra_strong_forward_pair_visits":step*32,
                        "total_loss":sup_total+self.cfg["training"]["lambda_u"]*unsup_total+self.cfg["method_settings"]["ent_loss_ratio"]*float(fairness),
                        "lambda_u_effective": ramp * self.cfg["training"]["lambda_u"],
                        "util_ratio": float(mask.mean()), "pseudo_positive_fraction": float((target_u >= .5).float().mean()),
                        "seconds": seconds, "physical": self.physical, "scale": self.scaler.get_scale(),
                        "fp32_retry": fp32, "attempts": attempts, "l_pair_visits": step*32,
                        "u_pair_visits": step*32, "unique_u_pairs": len(self.u_seen), "updated_utc": now()}
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                lb = uw = us = weak = z = loss = raw_loss = inputs = target_u = mask = strong = fairness = fairness_grad = proposed = thresholds = None
                gc.collect()
                torch.cuda.empty_cache()
                if self.physical <= 4:
                    raise
                self.physical //= 2
                # Reset scaler stage when OOM follows a partially constructed pass.
                state = self.scaler.state_dict()
                self.scaler = torch.amp.GradScaler("cuda")
                self.scaler.load_state_dict(state)


def check(rid,require_CPU=False):
    p=profile(require_CPU);bind_zero(base,p);cfg,contract,prov,step=base.check(rid)
    if step is not None:
        saved=torch.load(ROOT/'outputs'/rid/'last.pt',map_location='cpu',weights_only=False)
        validate_payload(saved,cfg,prov)
    return cfg,contract,prov,step
def assert_tree(a,b,rtol=1e-6,atol=1e-7):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=rtol if a.is_floating_point() else 0,atol=atol if a.is_floating_point() else 0)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise RuntimeError('NumPy RNG differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise RuntimeError('State tree topology differs')
        for k in a:assert_tree(a[k],b[k],rtol,atol)
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise RuntimeError('State sequence differs')
        for x,y in zip(a,b):assert_tree(x,y,rtol,atol)
    elif a!=b:raise RuntimeError('Full state scalar differs')
def validate_payload(v,cfg,prov):
    keys={'format','provenance','step','target_steps','trainable_state','ema','optimizer','scaler','rng','physical','initial_state_sha256','best_key','best_step','best_metrics','elapsed_seconds','u_seen','wall_seconds','algorithm_state','algorithm_checkpoint_version'}
    if not keys<=set(v) or v['format']!='pair_ssl_full_v1' or v['provenance']!=prov or v['target_steps']!=2600 or v['initial_state_sha256']!=NAMED or v['algorithm_checkpoint_version']!='freematch_full_v1' or not 0<=v['step']<=2600:raise RuntimeError('Incomplete/different FreeMatch full checkpoint')
    if len(v['trainable_state'])!=34 or set(v['trainable_state'])!=set(v['ema']) or v['physical'] not in (16,8,4) or set(v['rng'])!={'python','numpy','torch','cuda'} or len(v['rng']['cuda'])!=1 or v['scaler'].get('scale',0)<=0:raise RuntimeError('Incomplete student/EMA/scaler/RNG')
    state=FreeMatchState();state.load(v['algorithm_state'])
    if state.updates!=v['step']:raise RuntimeError('SAT update count differs')
    if v['step'] and (len(v['optimizer']['state'])!=34 or any(not {'step','exp_avg','exp_avg_sq'}<=set(s) for s in v['optimizer']['state'].values())):raise RuntimeError('Incomplete optimizer')
    def finite(x):
        if torch.is_tensor(x):return not x.is_floating_point() or bool(torch.isfinite(x).all())
        if isinstance(x,np.ndarray):return not np.issubdtype(x.dtype,np.floating) or bool(np.isfinite(x).all())
        if isinstance(x,dict):return all(finite(i) for i in x.values())
        if isinstance(x,(list,tuple)):return all(finite(i) for i in x)
        return not isinstance(x,float) or math.isfinite(x)
    if not finite(v):raise RuntimeError('Nonfinite complete state')
def reconstruct_first_cache(runner):
    seed=runner.cfg['seed'];l=random.Random(fixed_seed(seed,1,'pair-sampler')).sample(runner.l,16);u=random.Random(fixed_seed(seed,1,'u-sampler')).sample(runner.u,32)
    for pairs,view in ((flatten_l(l),'weak'),(u,'weak'),(u,'strong')):runner.cache.batch(pairs,1,view,runner.physical)
def gate(rid):
    cfg,contract,prov,resumed=check(rid,True)
    if resumed is not None or any((ROOT/'outputs'/rid/n).exists() for n in ('result.json','status.json','best.pt')) or list((ROOT/'outputs'/rid).glob('train_*.jsonl')):raise RuntimeError('Engineering only before untouched formal group')
    folder=ROOT/'audit/gpu_gates'/rid
    if folder.exists():raise RuntimeError('Existing engineering attempt preserved; no implicit retry')
    folder.mkdir(parents=True);started=time.monotonic();runner=Runner(cfg,contract,prov)
    one=runner.train_step();first=runner.snapshot();atomic_torch(folder/'engineering_last1.pt',first)
    two=runner.train_step();reference=runner.snapshot();atomic_torch(folder/'reference2.pt',reference)
    del runner;gc.collect();torch.cuda.empty_cache()
    replay_runner=Runner(cfg,contract,prov);reconstruct_first_cache(replay_runner)
    replay_runner.load(torch.load(folder/'engineering_last1.pt',map_location='cpu',weights_only=False));replayed_record=replay_runner.train_step();replay=replay_runner.snapshot();atomic_torch(folder/'replayed2.pt',replay)
    for key in ('trainable_state','ema'):
        assert_tree(reference[key],replay[key],rtol=0,atol=1e-6)
    for key in ('optimizer','algorithm_state'):assert_tree(reference[key],replay[key])
    for key in ('scaler','rng','physical','step','target_steps','initial_state_sha256'):assert_tree(reference[key],replay[key],rtol=0,atol=0)
    excluded={'seconds','updated_utc'}
    for key in two.keys()-excluded:
        a,b=two[key],replayed_record[key]
        if isinstance(a,float):
            if abs(a-b)>1e-6:raise RuntimeError('Cold logical-step record differs '+key)
        elif a!=b:raise RuntimeError('Cold logical-step record differs '+key)
    peak=torch.cuda.max_memory_allocated();maxdiff=max(float((reference['trainable_state'][k]-replay['trainable_state'][k]).abs().max()) for k in reference['trainable_state'])
    receipt=dict(passed=True,provenance=prov,gpu_uuid=GPU,engineering_only=True,formal_successful_steps_added=0,no_Validation_Test=True,actual_engineering_updates=2,replayed_updates=1,full_student_EMA_optimizer_scaler_rng_SAT_SAF_replay=True,independent_public_step1_prefix_reconstruction=True,no_reference_cache_copy=True,one=one,two=two,replay_two=replayed_record,replay_max_parameter_difference=maxdiff,common_34_state_sha256=NAMED,common_34_raw_sha256=RAW,artifact_hashes={f.name:digest(f) for f in folder.glob('*.pt')},cuda_peak_allocated_bytes=peak,seconds=time.monotonic()-started,created_utc=now())
    atomic_json(folder/'passed.json',receipt);print(json.dumps(receipt),flush=True)
def completed(rid):
    cfg,contract,prov,_=check(rid,True);out=ROOT/'outputs'/rid;r=read(out/'result.json');s=read(out/'status.json')
    if r['run_id']!=rid or r['state']!='completed' or s['state']!='completed' or r['successful_steps']!=2600 or s['step']!=2600 or r['provenance']!=prov or r['test_evaluated']:raise RuntimeError('Result/status2600/source differs')
    for n in ('best','last'):
        f=out/(n+'.pt')
        if digest(f)!=r[n+'_sha256']:raise RuntimeError('Checkpoint hash differs')
        v=torch.load(f,map_location='cpu',weights_only=False);validate_payload(v,cfg,prov)
        if v['step']!=(r['best_step'] if n=='best' else 2600) or n=='best' and v['best_metrics']!=r['best_metrics']:raise RuntimeError('Checkpoint step/metrics differs')
    records=[json.loads(x) for f in sorted(out.glob('train_*.jsonl')) for x in f.read_text(encoding='utf-8').splitlines() if x.strip()]
    if [x['step'] for x in records]!=list(range(1,2601)) or any(x['algorithm_successful_updates']!=x['step'] for x in records):raise RuntimeError('Not continuous successful algorithm steps')
    if any(not math.isfinite(v) for row in records for v in row.values() if isinstance(v,float)):raise RuntimeError('Nonfinite successful-step record')
    for i in range(100,2601,100):
        val=read(out/('validation_%04d.json'%i));pred=read(out/('predictions_validation_%04d.json'%i))
        if val['step']!=i or val['evaluation_model']!='ema' or val['validation_anchors']!=400 or pred['split']!='validation' or pred['image_ids']!=[x['image_id'] for x in contract[4]] or len(pred['labels'])!=800 or len(pred['probabilities'])!=800:raise RuntimeError('Full stored fixed Validation differs')
    if read(out/('validation_%04d.json'%r['best_step']))!=r['best_metrics']:raise RuntimeError('Best Validation differs')
    return dict(run_id=rid,result=r,full_student_EMA_optimizer_scaler_rng_SAT_SAF=True,continuous_successful_steps=True,Validation_verified=True,all_output_hashes={f.name:digest(f) for f in out.iterdir() if f.is_file()},observed_utc=now())
base.config_for=config_for;base.Runner=Runner
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid')
    mode=parser.add_mutually_exclusive_group(required=True)
    for n in ('check','gate','run','verify-result'):mode.add_argument('--'+n,action='store_true')
    args=parser.parse_args()
    try:
        if args.check:
            cfg,contract,prov,step=check(args.run_id);print(json.dumps(dict(passed=True,run_id=args.run_id,resume_step=step,provenance=prov)))
        elif args.verify_result:print(json.dumps(completed(args.run_id)))
        else:
            if args.gpu_uuid!=GPU or torch.cuda.device_count()!=1:raise RuntimeError('Only registered one RTX4090')
            with exclusive(ROOT/'locks'/(GPU+'.lock')),exclusive(ROOT/'locks'/(args.run_id+'.lock')):
                if args.gate:gate(args.run_id)
                else:
                    check(args.run_id,True)
                    if (ROOT/'outputs'/args.run_id/'last.pt').exists():raise RuntimeError('Formal recovery requires separate user authorization')
                    base.run(args.run_id,GPU)
    except BaseException:
        atomic_json(ROOT/'audit/failures'/(args.run_id+'_'+str(os.getpid())+'.json'),dict(error=traceback.format_exc(),mode=vars(args),automatic_formal_recovery_authorized=False,utc=now()));raise
if __name__=='__main__':main()
