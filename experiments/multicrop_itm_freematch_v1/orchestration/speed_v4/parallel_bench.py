"""Concurrent engineering only, still compares against independent serial reference."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import bench
# Distinct preserved attempt folder while execution/physics are exactly the selected candidate.
original=bench.SpeedRunner
crop,variant=sys.argv[1:3]
Parent=bench.train.Runner if variant=='reference' else original
class Selected(Parent):
 def __init__(self,cfg,contract,prov,**kwargs):
  if variant=='reference':super().__init__(cfg,contract,prov)
  else:super().__init__(cfg,contract,prov,checkpointing=variant=='retained_checkpoint',decoded_capacity=256)
bench.SpeedRunner=Selected
bench.benchmark(crop,'parallel_'+variant,24)
