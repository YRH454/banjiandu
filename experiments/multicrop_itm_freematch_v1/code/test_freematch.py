"""CPU-only SAT/SAF reference, global-gradient, same-step, checkpoint guards."""
import ast,copy,types,unittest
from pathlib import Path
import torch
import freematch_algorithm as a

def reference():
    root=Path(__file__).resolve().parents[1]/'references/usb/freematch'
    names={'torch':torch,'MaskingHook':object}
    code=ast.parse((root/'utils.py').read_text(encoding='utf-8'))
    nodes=[n for n in code.body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_USB_utils','exec'),names)
    code=ast.parse((root/'freematch.py').read_text(encoding='utf-8'))
    nodes=[n for n in code.body if isinstance(n,ast.FunctionDef) and n.name in ('entropy_loss','replace_inf_to_zero')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_USB_freematch','exec'),names)
    return names

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(20260825);torch.set_num_threads(2)
    def test_initial_uniform_binary(self):
        state=a.FreeMatchState();self.assertEqual(float(state.time_p),0.5);self.assertEqual(state.updates,0)
    def test_reference_SAT_multistep(self):
        ref=reference();hook=ref['FreeMatchThresholingHook'](2,momentum=0.999)
        algorithm=types.SimpleNamespace(distributed=False,world_size=1,use_quantile=False,clip_thresh=False)
        state=a.FreeMatchState()
        for step in range(1,8):
            z=torch.randn(32)*step;labels,mask,proposed,thresholds=state.propose(z)
            expected=hook.masking(algorithm,a.two_logits(z))
            torch.testing.assert_close(mask,expected,rtol=0,atol=0)
            for k in ('time_p','p_model','label_hist'):torch.testing.assert_close(proposed[k],getattr(hook,k),rtol=0,atol=0)
            state.commit(proposed);self.assertEqual(state.updates,step)
    def test_proposal_no_mutation_on_retry(self):
        state=a.FreeMatchState();original=state.state_dict()
        for _ in range(4):state.propose(torch.randn(32))
        self.assertEqual(state.updates,0)
        for k in ('time_p','p_model','label_hist'):torch.testing.assert_close(state.state_dict()[k],original[k],rtol=0,atol=0)
    def test_double_commit_rejected(self):
        s=a.FreeMatchState();_,_,p,_=s.propose(torch.zeros(32));s.commit(p)
        with self.assertRaises(RuntimeError):s.commit(p)
    def test_reference_SAF_value_and_gradient(self):
        ref=reference();state=a.FreeMatchState();_,mask,p,_=state.propose(torch.randn(32));z=torch.randn(32,requires_grad=True)
        got=a.saf_loss(z,mask,p);wanted,_=ref['entropy_loss'](mask,a.two_logits(z),p['p_model'],p['label_hist'])
        torch.testing.assert_close(got,wanted,rtol=0,atol=0)
        g=torch.autograd.grad(got,z,retain_graph=True)[0];w=torch.autograd.grad(wanted,z)[0]
        torch.testing.assert_close(g,w,rtol=1e-6,atol=1e-7)
    def test_global_two_pass_microbatch_gradient(self):
        state=a.FreeMatchState();_,mask,p,_=state.propose(torch.randn(32));base=torch.randn(32)
        z=base.clone().requires_grad_();full=a.saf_loss(z,mask,p);expected=torch.autograd.grad(full,z)[0]
        value,g=a.saf_value_and_gradient(base,mask,p)
        for physical in (16,8,4):
            pieces=[v.clone().requires_grad_() for v in base.split(physical)]
            surrogate=sum((v*g[i*physical:i*physical+len(v)]).sum() for i,v in enumerate(pieces))
            got=torch.cat(torch.autograd.grad(surrogate,pieces))
            torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(value,full.detach(),rtol=0,atol=0)
    def test_empty_mask_zero_value_gradient(self):
        s=a.FreeMatchState();v,g=a.saf_value_and_gradient(torch.randn(32),torch.zeros(32),s.state_dict());self.assertEqual(float(v),0);self.assertEqual(float(g.abs().sum()),0)
    def test_absent_strong_class_finite(self):
        s=a.FreeMatchState();v,g=a.saf_value_and_gradient(torch.full((32,),5.0),torch.ones(32),s.state_dict());self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(g).all())
    def test_requires_all_logical_U(self):
        with self.assertRaises(RuntimeError):a.FreeMatchState().propose(torch.zeros(16))
    def test_full_checkpoint_state_roundtrip(self):
        s=a.FreeMatchState();_,_,p,_=s.propose(torch.randn(32));s.commit(p);r=a.FreeMatchState();r.load(s.state_dict())
        for k in ('time_p','p_model','label_hist'):torch.testing.assert_close(getattr(s,k),getattr(r,k),rtol=0,atol=0)
        self.assertEqual(r.updates,1)
    def test_incomplete_checkpoint_refused(self):
        s=a.FreeMatchState();p=s.state_dict();del p['label_hist']
        with self.assertRaises(RuntimeError):s.load(p)
    def test_nonfinite_checkpoint_refused(self):
        s=a.FreeMatchState();p=s.state_dict();p['time_p']=torch.tensor(float('nan'))
        with self.assertRaises(RuntimeError):s.load(p)
    def test_invalid_momentum_refused(self):
        with self.assertRaises(ValueError):a.FreeMatchState(0.99)
    def test_ce_BCE_binary_equivalence(self):
        z=torch.randn(32);labels=torch.randint(0,2,(32,))
        torch.testing.assert_close(torch.nn.functional.cross_entropy(a.two_logits(z),labels,reduction='none'),torch.nn.functional.binary_cross_entropy_with_logits(z,labels.float(),reduction='none'),rtol=1e-6,atol=1e-7)
if __name__=='__main__':unittest.main(verbosity=2)
