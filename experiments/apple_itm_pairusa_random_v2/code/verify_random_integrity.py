"""Independent structural checks and in-memory tamper tests; no GPU training."""
import csv
import hashlib
import io
import json
from pathlib import Path
import train_queue as queue

ROOT=Path(__file__).resolve().parents[1]

class MemoryCSV:
    def __init__(self,rows):
        self.text=io.StringIO(newline='')
        w=csv.DictWriter(self.text,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    def open(self,*args,**kwargs):return io.StringIO(self.text.getvalue())
    def __str__(self):return '<synthetic in-memory tamper case>'

def main():
    manifest=queue.load_manifest(require_gate=True)
    previous={};counts={};first=None
    for code in ('001','005','010','020','030','100'):
        rows=queue.read_pairs(ROOT/manifest['files']['budgets'][code])
        allowed=queue.source_ids(queue.V2_DATA/f'budgets/train_{code}.csv')
        assert {r['image_id'] for r in rows}==allowed
        now={r['image_id']:r['negative_text_sha256'] for r in rows}
        assert all(now[k]==v for k,v in previous.items())
        previous=now;counts[code]=len(rows)
        if first is None:first=rows
    val=queue.read_pairs(ROOT/manifest['files']['validation'])
    assert len(val)==400
    assert {r['image_id'] for r in val}<=queue.source_ids(queue.V2_DATA/'validation.csv')
    assert not set(previous)&{r['image_id'] for r in val}
    tests={}
    for name in ('self_pair','outside_donor','edited_negative','forged_visual_status','wrong_positive_hash'):
        rows=[dict(r) for r in first]
        row=rows[0]
        if name=='self_pair':row['negative_source_image_id']=row['image_id']
        if name=='outside_donor':row['negative_source_image_id']='not_in_this_budget.jpg'
        if name=='edited_negative':
            row['negative_text']+=' Synthetic tamper.'
            row['negative_text_sha256']=hashlib.sha256(row['negative_text'].encode()).hexdigest()
        if name=='forged_visual_status':row['review_status']='approved'
        if name=='wrong_positive_hash':row['source_text_sha256']='0'*64
        try:queue.read_pairs(MemoryCSV(rows))
        except ValueError:tests[name]='rejected'
        else:raise AssertionError(f'Tampering was accepted: {name}')
    report={'passed':True,'scope':'all pair files read-only plus in-memory tamper cases',
            'train_counts':counts,'validation_count':len(val),'overlap_mapping_stable':True,
            'tamper_tests':tests,'semantic_visual_certification':False,'gpu_training_performed':False}
    path=ROOT/'audit/random_integrity_checks.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
