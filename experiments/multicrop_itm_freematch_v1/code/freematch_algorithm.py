"""FreeMatch SAT+SAF, USB 1ef4cbeb/ICLR2023 Eq5-12, binary pair-ITM adapter.

Statistics are proposed once per logical32 U batch and committed only after a
successful optimizer update. SAF is evaluated over the whole selected batch,
never independently per physical microbatch. No class labels for U are read.
"""
import torch
from torch.nn import functional as F

def two_logits(logits):
    return torch.stack((torch.zeros_like(logits),logits),dim=-1)

class FreeMatchState:
    def __init__(self,momentum=0.999,device='cpu'):
        if momentum!=0.999:raise ValueError('Registered EMA momentum is 0.999')
        self.momentum=momentum
        self.time_p=torch.tensor(0.5,device=device)
        self.p_model=torch.full((2,),0.5,device=device)
        self.label_hist=torch.full((2,),0.5,device=device)
        self.updates=0

    def state_dict(self):
        return dict(version='freematch_sat_saf_binary_v1',momentum=self.momentum,
                    time_p=self.time_p.detach().cpu().clone(),p_model=self.p_model.detach().cpu().clone(),
                    label_hist=self.label_hist.detach().cpu().clone(),updates=self.updates)

    def load(self,state):
        required={'version','momentum','time_p','p_model','label_hist','updates'}
        if set(state)!=required or state['version']!='freematch_sat_saf_binary_v1' or state['momentum']!=self.momentum:
            raise RuntimeError('Incomplete/different FreeMatch state')
        if not isinstance(state['updates'],int) or state['updates']<0:raise RuntimeError('Invalid SAT successful-update count')
        for key,shape in (('time_p',()),('p_model',(2,)),('label_hist',(2,))):
            v=state[key]
            if not torch.is_tensor(v) or tuple(v.shape)!=shape or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) or bool((v<=0).any()) or bool((v>1).any()):
                raise RuntimeError('Invalid FreeMatch statistic '+key)
            setattr(self,key,v.detach().to(self.time_p.device).clone())
        self.updates=state['updates']

    @torch.no_grad()
    def propose(self,weak_logits):
        if weak_logits.shape!=(32,) or not bool(torch.isfinite(weak_logits).all()):raise RuntimeError('SAT requires all32 finite U logits')
        probs=two_logits(weak_logits.detach().float()).softmax(dim=-1)
        confidence,pseudo=probs.max(dim=-1)
        m=self.momentum
        next_time=self.time_p*m+(1-m)*confidence.mean()
        next_model=self.p_model*m+(1-m)*probs.mean(dim=0)
        hist=torch.bincount(pseudo,minlength=2).to(probs.dtype)
        next_hist=self.label_hist*m+(1-m)*(hist/hist.sum())
        thresholds=next_time*next_model/next_model.max()
        mask=confidence.ge(thresholds[pseudo]).to(probs.dtype)
        proposed=dict(version='freematch_sat_saf_binary_v1',momentum=m,time_p=next_time,
                      p_model=next_model,label_hist=next_hist,updates=self.updates+1)
        return pseudo.to(probs.dtype),mask,proposed,thresholds

    def commit(self,proposed):
        if proposed['updates']!=self.updates+1:raise RuntimeError('SAT must commit exactly once after a successful step')
        self.load(proposed)

def saf_loss(strong_logits,mask,state):
    """Exact USB entropy_loss sign, masked histogram modulation, epsilon1e-12."""
    if strong_logits.shape!=(32,) or mask.shape!=(32,):raise RuntimeError('SAF requires global32 U batch')
    selected=mask.bool()
    if not bool(selected.any()):return strong_logits.sum()*0.0
    logits=two_logits(strong_logits.float())[selected]
    probs=logits.softmax(dim=-1)
    hist=torch.bincount(probs.argmax(dim=-1),minlength=2).to(probs.dtype)
    hist=hist/hist.sum()
    prior=state['p_model'].reshape(1,-1)
    counts=state['label_hist'].reshape(1,-1)
    scale=torch.reciprocal(counts).detach()
    scale=torch.where(torch.isinf(scale),torch.zeros_like(scale),scale)
    target=prior*scale;target=target/target.sum(dim=-1,keepdim=True)
    scale_s=torch.reciprocal(hist).detach()
    scale_s=torch.where(torch.isinf(scale_s),torch.zeros_like(scale_s),scale_s)
    mod=probs.mean(dim=0,keepdim=True)*scale_s
    mod=mod/mod.sum(dim=-1,keepdim=True)
    return (target*torch.log(mod+1e-12)).sum(dim=1).mean()

def saf_value_and_gradient(strong_logits,mask,state):
    """Two-pass exact chain-rule gradient of GLOBAL SAF for bounded GPU memory.

    The registered model has dropout disabled. The scientific runner must check
    that its physical replay logits equal these first-pass logits bitwise.
    This extra strong forward's actual compute/time must be reported.
    """
    with torch.enable_grad():
        detached=strong_logits.detach().float().requires_grad_(True)
        value=saf_loss(detached,mask,state)
        gradient=torch.autograd.grad(value,detached)[0]
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):raise FloatingPointError('Nonfinite global SAF')
    return value.detach(),gradient.detach()
