"""Original zero-step common34 template, exact hashes, no trained weight/teacher."""
import hashlib
from common import ROOT,digest
NAMED='b42eca74de637ed54ecaeea5268645a42dc0a9931ee45a2917591d4a1848704c'
RAW='3dd177e33a5a7f2f3ac7e43dd8a28236bdc4d20118af3df1381a4752a98ace74'
def bind_zero(base,profile):
    if getattr(base,'_fifth_zero_bound',False):return
    meta=profile['common34_zero'];path=ROOT/meta['path']
    if digest(path)!=meta['sha256'] or path.stat().st_size!=meta['bytes']:raise RuntimeError('Zero template file changed')
    before=base.torch.get_rng_state().clone();v=base.torch.load(path,map_location='cpu',weights_only=False)
    if v['format']!='portable_registered_common34_zero_v2' or v['successful_steps']!=0 or v['optimizer_updates']!=0 or v['loaded_BCE_best_or_teacher'] is not False or v['seed']!=20260825 or v['source']!=profile['common34_reference_source'] or v['official_ALBEF4M_sha256']!=digest(ROOT/'assets/ALBEF_4M.pth'):raise RuntimeError('Not original untrained public34')
    state=v['trainable_state'];raw=hashlib.sha256(b''.join(x.detach().cpu().contiguous().numpy().tobytes() for _,x in sorted(state.items()))).hexdigest()
    if len(state)!=34 or base.common_digest(state)!=NAMED or raw!=RAW:raise RuntimeError('Exact common34 values differ')
    if not base.torch.equal(before,base.torch.get_rng_state()):raise RuntimeError('Validation changed RNG')
    original=base.PairITMModel
    class ExactZero(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);rng=base.torch.get_rng_state().clone();self.load_trainable_state(state)
            actual=self.trainable_state()
            if any(not base.torch.equal(actual[k],state[k]) for k in state) or not base.torch.equal(rng,base.torch.get_rng_state()):raise RuntimeError('Exact zero copy/topology/RNG differs')
    base.PairITMModel=ExactZero
    prior=base.provenance
    def provenance(cfg,manifest):
        return {**prior(cfg,manifest),'fifth_execution_profile_sha256':digest(ROOT/'configs/execution_profile.json'),'common34_zero_template_sha256':meta['sha256'],'method':'FreeMatch_pair_ITM_SAT_SAF_v1'}
    base.provenance=provenance;base._fifth_zero_bound=True
