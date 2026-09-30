"""Build the user-authorized, full-caption random-other-case baseline.

No semantic/visual certainty is claimed. Captions are copied verbatim, not
parsed or rewritten. Each nested budget is a union of immutable derangements
within successive increments of its legally visible labelled image set.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'apple_multimodal_ce_bce_v2/data'
POLICY = 'random_other_case_full_caption_v2'
SEED = 20260825
BUDGETS = ('001','005','010','020','030','100')
EXPECTED = (133,668,1337,2674,4011,13373)
LABELS = ('healthy','scab','rust','frog_eye_leaf_spot','powdery_mildew','complex')
FIELDS = ('image_id','image_relpath','positive_text','negative_text','negative_source_image_id',
          'negative_source_image_relpath','edit_type','review_status','rule_id',
          'source_text_sha256','negative_text_sha256','image_sha256','negative_source_image_sha256',
          'first_visible_budget','donor_cohort','candidate_reason',*LABELS)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def text_hash(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()

def write_new(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError(f'Refusing to overwrite registered artifact: {path}')
        return
    with path.open('xb') as f:f.write(data)

def json_new(path,value):
    write_new(path,(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'))

def csv_new(path,rows):
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
    write_new(path,out.getvalue().encode('utf-8'))

def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def allowed_captions(wanted):
    wanted=set(wanted);result={}
    key=re.compile(r'^\s*"([^\"]+)"\s*:\s*(.+?),?\s*$')
    with (SOURCE/'blind_captions.json').open(encoding='utf-8') as f:
        for line in f:
            match=key.match(line)
            if not match or match.group(1) not in wanted:continue
            value=json.loads(match.group(2).rstrip(','))
            if not isinstance(value,str) or not value.strip():raise ValueError('Empty/nonstring caption')
            result[match.group(1)]=value
    if set(result)!=wanted:raise ValueError('Missing accessible captions')
    return result

def image_path(row):
    path=(SOURCE/row['image_relpath']).resolve()
    if SOURCE.resolve() not in path.parents or not path.is_file():raise ValueError('Invalid image path')
    return path

def stable_seed(tag):return int(text_hash(f'{SEED}\0{tag}')[:16],16)

def make_cohort(rows,cohort,tokenizer):
    rows=sorted(rows,key=lambda r:r['image_id'])
    captions=allowed_captions(r['image_id'] for r in rows)
    texts=[captions[r['image_id']] for r in rows]
    lengths=[len(ids) for ids in tokenizer(texts,padding=False,truncation=False)['input_ids']]
    if any(n>256 for n in lengths):
        raise ValueError(f'{cohort}: {sum(n>256 for n in lengths)} original captions exceed 256 tokens; not silently truncating')
    with ThreadPoolExecutor(max_workers=8) as pool:
        image_hashes=list(pool.map(digest,[image_path(r) for r in rows]))
    n=len(rows)
    if n<2:raise ValueError('Random-negative cohort needs at least two cases')
    rng=random.Random(stable_seed(f'derangement/{cohort}'))
    permutation=list(range(n))
    for attempt in range(10000):
        rng.shuffle(permutation)
        if all(i!=j and texts[i]!=texts[j] and image_hashes[i]!=image_hashes[j] for i,j in enumerate(permutation)):
            break
    else:raise RuntimeError(f'Could not create a valid random derangement for {cohort}')
    paired=[]
    for i,j in enumerate(permutation):
        row,donor=rows[i],rows[j]
        pair={'image_id':row['image_id'],'image_relpath':row['image_relpath'],
              'positive_text':texts[i],'negative_text':texts[j],
              'negative_source_image_id':donor['image_id'],'negative_source_image_relpath':donor['image_relpath'],
              'edit_type':'random_other_case_full_caption','review_status':'random_other_case_unverified',
              'rule_id':POLICY,'source_text_sha256':text_hash(texts[i]),'negative_text_sha256':text_hash(texts[j]),
              'image_sha256':image_hashes[i],'negative_source_image_sha256':image_hashes[j],
              'first_visible_budget':cohort,'donor_cohort':cohort,
              'candidate_reason':'User-authorized identity mismatch; original caption copied verbatim; semantic false negatives not screened.'}
        pair.update({k:row[k] for k in LABELS});paired.append(pair)
    return paired,{'cohort':cohort,'anchors':n,'shuffle_attempts':attempt+1,
                   'tokens_min':min(lengths),'tokens_max':max(lengths),'unique_captions':len(set(texts)),
                   'unique_image_files':len(set(image_hashes))}

def verify_rows(rows,expected_ids):
    by_id={r['image_id']:r for r in rows}
    assert len(by_id)==len(rows) and set(by_id)==set(expected_ids)
    assert Counter(r['positive_text'] for r in rows)==Counter(r['negative_text'] for r in rows)
    assert Counter(r['negative_source_image_id'] for r in rows)==Counter(by_id.keys())
    for row in rows:
        donor=by_id[row['negative_source_image_id']]
        assert donor['image_id']!=row['image_id']
        assert donor['positive_text']==row['negative_text']!=row['positive_text']
        assert row['image_sha256']!=row['negative_source_image_sha256']==donor['image_sha256']
        assert text_hash(row['positive_text'])==row['source_text_sha256']
        assert text_hash(row['negative_text'])==row['negative_text_sha256']
        assert row['review_status']=='random_other_case_unverified'

def verify_caption_only(rows):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    labels=np.array([1]*len(rows)+[0]*len(rows))
    texts=[r['positive_text'] for r in rows]+[r['negative_text'] for r in rows]
    # Arbitrary deterministic text scoring, deliberately ignoring images.
    scores=np.array([int(text_hash(t)[:12],16) for t in texts],dtype=np.float64)
    auc=float(roc_auc_score(labels,scores))
    assert abs(auc-.5)<1e-12
    return {'same_caption_multiset_both_labels':True,'hash_text_score_auroc':auc,
            'deterministic_caption_only_auroc_by_identity':.5,
            'not_a_visual_correctness_certificate':True}

def main():
    manifest_path=ROOT/'data/manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        gate=json.loads((ROOT/'audit/negative_quality_gate.json').read_text(encoding='utf-8'))
        assert manifest['negative_policy']==gate['negative_policy']==POLICY
        for relative,expected in gate['admitted_file_sha256'].items():assert digest(ROOT/relative)==expected
        print('Existing immutable manifests verified; not regenerated.');return
    from transformers import BertTokenizerFast
    tokenizer=BertTokenizerFast.from_pretrained('C:/Users/Lenovo/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594',local_files_only=True)
    start=time.time();prior=set();pairs_by_id={};source_hashes={};cohorts=[];files={};pair_hashes={};counts={}
    previous_negatives={}
    for code,expected in zip(BUDGETS,EXPECTED):
        path=SOURCE/f'budgets/train_{code}.csv';source_hashes[str(path)]=digest(path)
        source=read_rows(path);ids={r['image_id'] for r in source}
        assert len(source)==len(ids)==expected and prior<=ids
        paired,record=make_cohort([r for r in source if r['image_id'] not in prior],code,tokenizer)
        cohorts.append(record);pairs_by_id.update({r['image_id']:r for r in paired})
        rows=[pairs_by_id[key] for key in sorted(ids)]
        verify_rows(rows,ids)
        for image_id,text in previous_negatives.items():assert pairs_by_id[image_id]['negative_text']==text
        previous_negatives={r['image_id']:r['negative_text'] for r in rows}
        relative=f'data/pairs/train_{code}.csv';csv_new(ROOT/relative,rows)
        files[code]=relative;pair_hashes[relative]=digest(ROOT/relative);counts[code]=len(rows)
        prior=ids
        print(json.dumps({'budget':code,'anchors':len(rows),'cohort':record,'caption_probe':verify_caption_only(rows)},ensure_ascii=False),flush=True)
    val_source=read_rows(SOURCE/'validation.csv');val_ids={r['image_id'] for r in val_source}
    test_ids={r['image_id'] for r in read_rows(SOURCE/'test.csv')}
    assert not (prior&val_ids or prior&test_ids or val_ids&test_ids)
    selected=random.Random(stable_seed('validation-400-selection')).sample(sorted(val_source,key=lambda r:r['image_id']),400)
    val_rows,val_record=make_cohort(selected,'validation400',tokenizer)
    val_rows.sort(key=lambda r:r['image_id']);verify_rows(val_rows,{r['image_id'] for r in selected})
    relative='data/pairs/validation.csv';csv_new(ROOT/relative,val_rows);pair_hashes[relative]=digest(ROOT/relative)
    source_hashes[str(SOURCE/'validation.csv')]=digest(SOURCE/'validation.csv')
    report={'policy':POLICY,'seed':SEED,'created_unix':time.time(),'seconds':time.time()-start,
            'train_counts':counts,'cohorts':cohorts,'validation':val_record,
            'caption_only_validation_structural_check':verify_caption_only(val_rows),
            'nested_pairs_stable':True,'donor_ids_within_own_budget_or_validation_subset':True,
            'self_or_identical_text_or_identical_file_pairs':0,'all_captions_verbatim':True,
            'semantic_visual_review_performed':False,'semantic_false_negative_rate':'not measured',
            'test_captions_read':False,'test_evaluation':False,'source_csv_sha256':source_hashes,
            'generator_sha256':digest(Path(__file__)),'pair_file_sha256':pair_hashes}
    json_new(ROOT/'audit/random_pair_preparation.json',report)
    gate={'ready_for_training':True,'negative_policy':POLICY,'stage':'structural_checks_passed_for_authorized_random_policy',
          'scientific_stop_requires_user_decision':False,'reasons':[],
          'semantic_visual_approval':False,'semantic_false_negatives_possible':True,
          'authorization':'reports/执行登记_v5_随机异病例完整caption.md',
          'admitted_file_sha256':pair_hashes,'formal_training_started':False,
          'scope':'Structural provenance, random pairing and text-marginal checks only; no gold-standard semantic claim.'}
    json_new(ROOT/'audit/negative_quality_gate.json',gate)
    manifest={'version':2,'negative_policy':POLICY,'seed':SEED,'image_root':str(SOURCE),
              'files':{'budgets':files,'validation':'data/pairs/validation.csv'},
              'source_counts':dict(zip(BUDGETS,EXPECTED)),'validation_anchors':400,
              'randomization':'fixed derangements inside first-visible nested-budget cohorts',
              'caption_status':'verbatim blind captions; random negatives semantically unverified'}
    json_new(manifest_path,manifest)
    print(json.dumps({'completed':True,'train_counts':counts,'validation_anchors':400,'seconds':time.time()-start},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
