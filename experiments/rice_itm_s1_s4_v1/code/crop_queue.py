"""Versioned rice/banana adapter for the registered cassava fast mathematics.

--check is CPU-only. --run refuses to use the GPU until its predecessor is
complete and the shared cassava GPU lock is exclusively acquired. Engineering
GPU gates live in audit, never count as formal student results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / 'configs/plan.json').read_text(encoding='utf-8'))
MASTER = Path(PLAN['master_root'])
sys.path.insert(0, str(MASTER / 'code'))
from common import exclusive_lock, read, ready, verify_completed

import ot_stage_fast as ot
sys.modules['ot_stage'] = ot
import cassava_queue as q
from albef_ssl.model import get_tokenizer

ORIGINAL_MODEL_CONFIG = q.base.model_config
ORIGINAL_BASE_CONFIG = q.base_config
ORIGINAL_OT_CONFIG = q.ot_config
ORIGINAL_U_LOADER = ot.load_u_rows


def load_crop_u_rows(path, expected_hash=None):
    if PLAN['dataset'] != 'banana':
        return ORIGINAL_U_LOADER(path, expected_hash)
    path = Path(path)
    if expected_hash and q.digest(path) != expected_hash:
        raise RuntimeError('U table fingerprint changed')
    fields = ['pair_id', 'image_id', 'image_path', 'text', 'text_sha256']
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise RuntimeError('U loader rejects labels, donors and additional fields')
        rows = list(reader)
    registered = {r['image_id']: r['relpath'] for r in (json.loads(line) for line in
        (ROOT / 'data/image_index.jsonl').read_text(encoding='utf-8').splitlines())}
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate U pair identifiers')
    for row in rows:
        relative = Path(row['image_path'])
        if relative.is_absolute() or '..' in relative.parts or registered.get(row['image_id']) != row['image_path']:
            raise RuntimeError('U path is not the registered category-qualified banana image')
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise RuntimeError('U verbatim caption hash mismatch')
    return rows


def run_id(stage, budget):
    return f"{PLAN['dataset']}_itm_{stage.lower()}_l{budget}_s{q.SEED}"


def source_fingerprint(manifest):
    files = [ROOT / 'code' / name for name in ('crop_queue.py', 'legacy_parent.py', 'pair_model.py',
        'prepare_pairs.py', 'cassava_queue.py', 'ot_stage_fast.py', 'ot_loss.py', 'test_ot_loss.py')]
    files += [ROOT / rel for rel in ('configs/plan.json', 'configs/queue.json', 'reports/执行登记.md',
        'data/manifest.json', 'data/u_manifest.json', 'data/source_snapshot.json', 'data/source_file_hashes.jsonl',
        'data/image_index.jsonl', 'data/splits/split_manifest.json',
        'audit/source_audit.json', 'audit/negative_quality_gate.json', 'audit/pair_preparation.json',
        'audit/backend_copy_receipt.json', 'audit/ot_numerical_checks.json',
        'audit/ot_failure_20260930/legacy_ot_loss.py', 'audit/ot_failure_20260930/failure_inputs_step140.pt')]
    files += [ROOT / f'data/splits/{name}_ids.txt' for name in ('train', 'validation', 'test')]
    files += [ROOT / 'data/train_pool_ids.csv', ROOT / 'data/validation.csv', ROOT / 'data/test.csv']
    files += [ROOT / path for path in manifest['files']['budgets'].values()]
    files += [ROOT / manifest['files']['validation']]
    files += [ROOT / item['path'] for item in read(ROOT / 'data/u_manifest.json')['budgets'].values()]
    files += [ROOT / f'data/budgets/train_{b}.csv' for b in q.BUDGETS]
    files += [MASTER / 'code' / name for name in ('common.py', 'prepare_dataset.py', 'followup_queue.py')]
    files += [MASTER / 'configs/queue.json', MASTER / 'reports/后续队列执行登记.md']
    files += [q.base.CORE / rel for rel in ('albef_ssl/model.py', 'albef_ssl/vendor/albef/vit.py',
        'albef_ssl/vendor/albef/xbert.py', 'albef_ssl/vendor/albef/bert_config.json')]
    result = {str(path.relative_to(ROOT)).replace('\\', '/') if path.is_relative_to(ROOT) else str(path):
        q.digest(path) for path in files}
    result['source_revision'] = PLAN['source_revision']
    result['ALBEF_4M_md5'] = q.base.md5(Path(q.base.model_config()['checkpoint']))
    result['tokenizer_snapshot'] = q.base.path_fingerprint(Path(q.base.model_config()['tokenizer_path']))
    return result


def configure():
    def model_config():
        cfg = ORIGINAL_MODEL_CONFIG()
        cfg['max_text_length'] = PLAN['max_text_tokens_including_special_tokens']
        cfg['fusion_chunk_size'] = 16
        return cfg

    q.COUNT = PLAN['l_counts']
    q.BUDGETS = tuple(PLAN['budget_order'])
    q.run_id = run_id
    q.ot = ot
    ot.load_u_rows = load_crop_u_rows
    q.OT_CONFIG = {**q.OT_CONFIG, 'tolerance': 1e-4}
    q.source_fingerprint = source_fingerprint
    for backend in (q.base, ot.parent):
        backend.EXPECTED_COUNTS = PLAN['l_counts']
        backend.model_config = model_config
        backend.source_fingerprint = source_fingerprint

    def base_config(manifest, budget, stage):
        cfg = ORIGINAL_BASE_CONFIG(manifest, budget, stage)
        cfg['model']['max_text_length'] = PLAN['max_text_tokens_including_special_tokens']
        cfg['model']['fusion_chunk_size'] = 16
        cfg['speed_profile'] = PLAN['speed_profile']
        cfg['recovery_schema'] = 'crop_full_state_v1'
        return cfg

    def ot_config(manifest, fp, budget, stage):
        cfg = ORIGINAL_OT_CONFIG(manifest, fp, budget, stage)
        cfg['model']['fusion_chunk_size'] = 16
        cfg['ot'] = {**cfg['ot'], 'tolerance': 1e-4}
        cfg['speed_profile'] = PLAN['speed_profile']
        cfg['recovery_schema'] = 'crop_full_state_v1'
        return cfg

    q.base_config, q.ot_config = base_config, ot_config


configure()


def check_inputs(deep=False):
    expected = {'training_seed': 20260825, 'budget_order': ['005', '020', '001', '010', '030', '100'],
        'student_stages': 22, 'student_steps_per_stage': 1600, 'activation_checkpointing': True,
        'fusion_physical_microbatch': 16, 'ot_epsilon': .1, 'ot_tolerance': 1e-4, 'ot_max_iterations': 100,
        'logical_labeled_positive_anchors': 16, 'logical_unlabeled_pairs': 32,
        'experimental_speed_candidates_deployed': False, 'test_evaluation': False, 'caption_field': 'blind'}
    if any(PLAN.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Crop plan differs from the authorized cassava method')
    if PLAN['lineage'] != {'S1': 'ALBEF_4M', 'S2': 'same_budget_S1_best',
        'S3': 'same_budget_S1_best', 'S4': 'same_budget_S3_best'}:
        raise RuntimeError('Stage lineage changed')
    if PLAN['max_text_tokens_including_special_tokens'] != (400 if PLAN['dataset'] == 'rice' else 384):
        raise RuntimeError('Full-caption guard changed')
    manifest = q.base.load_manifest(require_gate=True)
    snapshot, audit = read(ROOT / 'data/source_snapshot.json'), read(ROOT / 'audit/source_audit.json')
    split, pairs = read(ROOT / 'data/splits/split_manifest.json'), read(ROOT / 'audit/pair_preparation.json')
    receipt = read(ROOT / 'audit/backend_copy_receipt.json')
    if (not audit['source_integrity_passed'] or not pairs['passed'] or not split['passed']
            or snapshot['source_revision'] != PLAN['source_revision'] or manifest['source_counts'] != q.COUNT
            or manifest['max_text_tokens'] != PLAN['max_text_tokens_including_special_tokens']
            or audit['source_snapshot_sha256'] != q.digest(ROOT / 'data/source_snapshot.json')
            or manifest['split_manifest_sha256'] != q.digest(ROOT / 'data/splits/split_manifest.json')
            or pairs['manifest_sha256'] != q.digest(ROOT / 'data/manifest.json')
            or pairs['u_manifest_sha256'] != q.digest(ROOT / 'data/u_manifest.json')
            or receipt['preparation_code_sha256'] != q.digest(MASTER / 'code/prepare_dataset.py')):
        raise RuntimeError('Crop source/split/pair binding failed')
    for name, item in receipt['files'].items():
        if q.digest(ROOT / 'code' / name) != item['local_sha256']:
            raise RuntimeError(f'Copied registered backend changed: {name}')
    bindings = [('metadata_path', 'manifest_sha256'), ('captions_path', 'captions_sha256'),
        ('download_audit_path', 'download_audit_sha256'), ('source_file_hashes_path', 'source_file_hashes_sha256'),
        ('remote_snapshot_path', 'remote_snapshot_sha256')]
    for path_key, hash_key in bindings:
        if q.digest(Path(snapshot[path_key])) != snapshot[hash_key]:
            raise RuntimeError(f'Frozen downloaded source changed: {path_key}')
    if q.digest(ROOT / 'data/source_file_hashes.jsonl') != snapshot['source_file_hashes_sha256']:
        raise RuntimeError('Frozen source file list changed')
    if deep:
        source = Path(PLAN['source_root'])
        for line in (ROOT / 'data/source_file_hashes.jsonl').read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            path = (source / item['path']).resolve()
            if not path.is_relative_to(source.resolve()) or path.stat().st_size != item['size'] or q.digest(path) != item['sha256']:
                raise RuntimeError(f'Downloaded payload changed: {path}')
    numerical = read(ROOT / 'audit/ot_numerical_checks.json')
    if not numerical['passed'] or numerical['check_count'] != 27 or any(
        numerical['source_sha256'][name] != q.digest(ROOT / 'code' / name) for name in ('ot_loss.py', 'test_ot_loss.py')):
        raise RuntimeError('Exact-source OT numerical gate missing')
    ids = {name: set((ROOT / f'data/splits/{name}_ids.txt').read_text(encoding='utf-8').splitlines())
           for name in ('train', 'validation', 'test')}
    if any(len(ids[name]) != PLAN['split_counts'][name] for name in ids) or any(
        ids[a] & ids[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise RuntimeError('Frozen split identity/counts changed')
    index = {r['image_id']: r for r in (json.loads(line) for line in
        (ROOT / 'data/image_index.jsonl').read_text(encoding='utf-8').splitlines())}
    if set(index) != set.union(*ids.values()) or split['index_sha256'] != q.digest(ROOT / 'data/image_index.jsonl'):
        raise RuntimeError('Index/split binding changed')
    groups = {}
    for name, members in ids.items():
        for identity in members:
            group = index[identity]['split_group_id']
            if group in groups and groups[group] != name:
                raise RuntimeError('Scene split group crosses pools')
            groups[group] = name
    val_rows = q.base.read_pairs(ROOT / manifest['files']['validation'])
    if len(val_rows) != 400 or not {r['image_id'] for r in val_rows} <= ids['validation']:
        raise RuntimeError('Fixed validation selection changed')
    u_manifest = read(ROOT / 'data/u_manifest.json')
    if u_manifest['source_pair_manifest_sha256'] != q.digest(ROOT / 'data/manifest.json') or u_manifest['max_text_tokens'] != manifest['max_text_tokens']:
        raise RuntimeError('U input registration changed')
    previous = set()
    previous_pairs = {}
    for budget in ('001', '005', '010', '020', '030', '100'):
        rows = q.base.read_pairs(ROOT / manifest['files']['budgets'][budget])
        allowed = {r['image_id'] for r in rows}
        if len(rows) != q.COUNT[budget] or allowed != q.base.source_ids(ROOT / f'data/budgets/train_{budget}.csv') or not previous <= allowed <= ids['train']:
            raise RuntimeError('Nested L IDs changed')
        for row in rows:
            if row['source_text_sha256'] != index[row['image_id']]['blind_sha256'] or row['image_relpath'] != index[row['image_id']]['relpath']:
                raise RuntimeError('L blind caption/path changed')
            if row['image_id'] in previous_pairs and row != previous_pairs[row['image_id']]:
                raise RuntimeError('L pair identity changed across nested budgets')
        previous, previous_pairs = allowed, {r['image_id']: r for r in rows}
        if budget != '100':
            item = u_manifest['budgets'][budget]
            u = ot.load_u_rows(ROOT / item['path'], item['sha256'])
            u_ids = ids['train'] - allowed
            if item['n_images'] != len(u_ids) or item['n_pairs'] != 2 * len(u_ids) or {r['image_id'] for r in u} != u_ids:
                raise RuntimeError('U partition cardinality changed')
            allowed_text = {index[identity]['blind_sha256'] for identity in u_ids}
            if any(r['text_sha256'] not in allowed_text or r['image_path'] != index[r['image_id']]['relpath'] for r in u):
                raise RuntimeError('U text/image crosses its frozen budget or uses guided captions')
    model = q.base.model_config()
    if q.base.md5(Path(model['checkpoint'])) != model['checkpoint_md5_expected']:
        raise RuntimeError('ALBEF_4M weight changed')
    fp = source_fingerprint(manifest)
    return manifest, fp


def gpu_gates(manifest, fp):
    path = ROOT / 'audit/gpu_training_verification.json'
    if path.exists():
        proof = read(path)
        if proof.get('passed') is not True or proof['source_fingerprint'] != fp:
            raise RuntimeError('Previous GPU engineering gate differs; do not overwrite')
        return proof
    if q.base.DEVICE.type != 'cuda':
        raise RuntimeError('CUDA is required for genuine engineering gates')
    tokenizer = get_tokenizer(q.base.model_config()['tokenizer_path'])
    rows = q.base.read_pairs(ROOT / manifest['files']['budgets']['100'])
    val_rows = q.base.read_pairs(ROOT / manifest['files']['validation'])
    ranked = []
    for begin in range(0, len(rows), 128):
        group = rows[begin:begin + 128]
        lengths = tokenizer([r['positive_text'] for r in group], truncation=False, padding=False)['input_ids']
        ranked.extend((len(ids), row) for ids, row in zip(lengths, group))
    selected = [r for _, r in sorted(ranked, key=lambda x: (-x[0], x[1]['image_id']))[:16]]
    engineering = ROOT / 'audit/gpu_gate'
    cfg = q.base_config(manifest, '100', 'S1')
    cfg['run_id'] = f"{PLAN['dataset']}_engineering_smoke_s1"
    cfg['engineering_only_not_formal_result'] = True
    cfg['selected_anchor_ids'] = [r['image_id'] for r in selected]
    cfg['training'] = {**cfg['training'], 'max_steps': 2, 'checkpoint_every': 1, 'eval_every': 2}
    original_outputs, original_save = q.base.OUTPUTS, q.base.atomic_torch
    q.base.OUTPUTS = engineering / 'outputs'

    class SimulatedInterruption(RuntimeError):
        pass

    injected = False
    def interrupted_save(target, value):
        nonlocal injected
        original_save(target, value)
        if Path(target).name == 'last.pt' and value.get('step') == 1 and not injected:
            injected = True
            raise SimulatedInterruption('engineering interruption after durable last1')

    try:
        folder = q.base.OUTPUTS / cfg['run_id']
        if not (folder / 'last.pt').exists():
            q.base.atomic_torch = interrupted_save
            try:
                q.base.run_student(cfg, selected, val_rows, tokenizer, fp)
            except SimulatedInterruption:
                pass
            finally:
                q.base.atomic_torch = original_save
        resumed = q.base.run_student(cfg, selected, val_rows, tokenizer, fp)
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if last['step'] != 2 or resumed['state'] != 'completed_validation' or len(last['trainable_state']) != 34:
            raise RuntimeError('Real S1 checkpoint did not advance after interruption')
        parent_best = engineering / 's1_real_update_parent.pt'
        original_save(parent_best, {'step': 2, 'trainable_state': last['trainable_state']})
    finally:
        q.base.atomic_torch = original_save
        q.base.OUTPUTS = original_outputs
    gc.collect()
    torch.cuda.empty_cache()

    budget = '005'
    u = read(ROOT / 'data/u_manifest.json')['budgets'][budget]
    cfg_ot = {'run_id': f"{PLAN['dataset']}_engineering_smoke_s3", 'variant': 'G3', 'stage': 'S3',
        'budget': budget, 'seed': q.SEED, 'model': {**q.base.model_config(), 'use_pairusa': False},
        'training': q.base.make_configs(manifest)[0]['training'],
        'pair_file': str(ROOT / manifest['files']['budgets'][budget]),
        'validation_file': str(ROOT / manifest['files']['validation']), 'image_root': manifest['image_root'],
        'u_file': str(ROOT / u['path']), 'u_sha256': u['sha256'], 'u_n_pairs': u['n_pairs'],
        'u_n_images': u['n_images'], 'unlabeled_batch': 32, 'max_cpu_prefix_cache_gib': 48,
        'warmstart': {'parent_kind': 'engineering_only_S1', 'parent_run_id': 'engineering_not_formal',
            'parent_best_path': str(parent_best), 'parent_best_sha256': q.digest(parent_best)},
        'ot': copy.deepcopy(q.OT_CONFIG), 'test_evaluation': False}
    runtime = ot.Runtime(cfg_ot, eager=False)
    record201 = ot.perform_step(runtime, 201)
    checkpoint = ot.capture_training_state(runtime)
    original_save(engineering / 'ot_replay_checkpoint.pt', checkpoint)
    record202 = ot.perform_step(runtime, 202)
    after = runtime.model.trainable_state()
    checkpoint = torch.load(engineering / 'ot_replay_checkpoint.pt', map_location='cpu', weights_only=False)
    ot.restore_training_state(runtime, checkpoint)
    replay = ot.perform_step(runtime, 202)
    max_error = max(float((after[key] - runtime.model.trainable_state()[key]).abs().max()) for key in after)
    if max_error > 1e-7 or record202['u_pair_ids'] != replay['u_pair_ids'] or not record201['diagnostics']['converged'] or not replay['diagnostics']['converged']:
        raise RuntimeError('OT real full-state 201->202 replay failed')
    shared_initial = runtime.model.trainable_state()
    usa_model = q.pair_model.PairITMModel({**q.base.model_config(), 'use_pairusa': True})
    if set(usa_model.trainable_state()) - set(shared_initial) != q.EXTRA:
        raise RuntimeError('USA extension is not exactly the five registered fresh keys')
    proof = {'passed': True, 'scope': 'engineering_only_not_formal_experiment',
        'source_fingerprint': fp, 'real_labeled_batch': 16, 'real_unlabeled_batch': 32,
        'max_training_caption_tokens': max(n for n, _ in ranked),
        'caption_guard': PLAN['max_text_tokens_including_special_tokens'],
        's1_interrupted_durable_step': 1, 's1_resumed_successful_step': 2,
        'ot_active_steps': [201, 202], 'ot_full_state_replayed_step': 202, 'ot_replay_max_tensor_error': max_error,
        'ot_diagnostics_201': record201['diagnostics'], 'ot_diagnostics_202': replay['diagnostics'],
        'usa_fresh_keys': sorted(q.EXTRA), 'common_keys': 34, 'usa_total_keys': 39, 'utc': q.base.now()}
    q.base.atomic_json(path, proof)
    del runtime, usa_model, checkpoint, after
    gc.collect()
    torch.cuda.empty_cache()
    return proof


def run(manifest, fp):
    predecessor = Path(PLAN['predecessor_root'])
    if not ready(predecessor, PLAN['predecessor_kind']):
        raise RuntimeError('Predecessor is still running/incomplete; refuse GPU work')
    # The exact parent cassava lock also protects all new crop GPU work.
    with exclusive_lock(Path(PLAN['cassava_gpu_lock'])), q.base.process_lock(), q.base.prevent_system_sleep():
        receipt = verify_completed(predecessor, PLAN['predecessor_kind'])
        q.immutable_json(ROOT / 'audit/predecessor_completion.json', receipt)
        path = ROOT / 'outputs/pipeline_state.json'
        ids = [run_id(stage, budget) for stage, budget in q.queue_order()]
        state = {'state': 'checking_real_gpu_gates', 'run_ids': ids, 'source_fingerprint': fp,
            'started_utc': q.base.now(), 'pid': __import__('os').getpid(), 'test_evaluation': False}
        if path.exists():
            previous = read(path)
            if previous['source_fingerprint'] != fp or previous['run_ids'] != ids:
                raise RuntimeError('Crop queue source/order differs; refusing resume')
            state['started_utc'] = previous['started_utc']
        q.base.atomic_json(path, state)
        try:
            torch.set_num_threads(8)
            torch.manual_seed(q.SEED); torch.cuda.manual_seed_all(q.SEED)
            np.random.seed(q.SEED % (2**32)); random.seed(q.SEED)
            torch.backends.cudnn.benchmark = True
            gpu_gates(manifest, fp)
            tokenizer = get_tokenizer(q.base.model_config()['tokenizer_path'])
            for stage, budget in q.queue_order():
                state.update(state='running', active_run=run_id(stage, budget), updated_utc=q.base.now())
                q.base.atomic_json(path, state)
                result = q.run_stage(manifest, fp, tokenizer, stage, budget)
                print(json.dumps({'completed': result['run_id'], 'best_step': result['best_step'],
                    'paired_accuracy': result['best_validation']['paired_accuracy']}), flush=True)
            state.update(state='completed_validation', active_run=None, completed_utc=q.base.now(), updated_utc=q.base.now())
            q.base.atomic_json(path, state)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc), updated_utc=q.base.now())
            q.base.atomic_json(path, state)
            q.base.atomic_json(ROOT / 'outputs/pipeline_failure.json',
                {'error': repr(exc), 'traceback': traceback.format_exc(), 'utc': q.base.now()})
            raise


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--deep-source-check', action='store_true')
    args = parser.parse_args()
    manifest, fp = check_inputs(deep=args.deep_source_check or args.run)
    if args.check:
        path = ROOT / 'outputs/pipeline_state.json'
        if path.exists() and read(path)['source_fingerprint'] != fp:
            previous = read(path)
            if previous['state'] != 'waiting_for_predecessor' or (ROOT / 'audit/gpu_training_verification.json').exists() or any(
                    (ROOT / 'outputs').glob('*_itm_*/last.pt')):
                raise RuntimeError('Registered executed queue fingerprint changed; no automatic migration')
            import shutil
            revision = ROOT / 'audit/preparation_backups' / f"cpu_adapter_{__import__('time').time_ns()}"
            revision.mkdir(parents=True)
            for old_path in (path, ROOT / 'audit/cpu_queue_verification.json'):
                if old_path.exists():
                    shutil.copy2(old_path, revision / old_path.name)
            q.base.atomic_json(revision / 'receipt.json', {'scope': 'unstarted CPU-only registration update',
                'reason': 'category-qualified banana U path adapter and pre-launch checks; no data/loss/model change',
                'previous_source_fingerprint': previous['source_fingerprint'], 'new_source_fingerprint': fp})
            q.base.atomic_json(path, {**previous, 'source_fingerprint': fp, 'updated_utc': q.base.now()})
        proof = {'passed': True, 'scope': 'cpu_static_contracts_only',
            'source_fingerprint': fp, 'student_stages': 22, 'gpu_gates_passed': False,
            'l_counts': q.COUNT, 'split_counts': PLAN['split_counts']}
        q.base.atomic_json(ROOT / 'audit/cpu_queue_verification.json', proof)
        if not path.exists():
            q.base.atomic_json(path, {'state': 'waiting_for_predecessor',
                'run_ids': [run_id(s, b) for s, b in q.queue_order()], 'source_fingerprint': fp,
                'started_utc': q.base.now(), 'test_evaluation': False, 'predecessor_root': PLAN['predecessor_root']})
        print(json.dumps({key: value for key, value in proof.items() if key != 'source_fingerprint'}, ensure_ascii=False))
    else:
        run(manifest, fp)


if __name__ == '__main__':
    main()
