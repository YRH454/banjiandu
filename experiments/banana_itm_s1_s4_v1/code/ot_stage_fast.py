# -*- coding: utf-8 -*-
"""Versioned fast OT backend for post-5% cassava stages only.

The registered logical batches and losses are unchanged.  A larger physical
fusion microbatch is allowed, with deterministic OOM backoff to 8 and 4.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT
C_ROOT = ROOT
OUTPUTS = ROOT / 'outputs'
REGISTRATION = ROOT / 'reports/木薯_S1-S4_单种子实验计划_v1_待审核.md'
SEED = 20260825
EXTRA_KEYS = frozenset({'student_projection.0.weight', 'student_projection.0.bias',
    'student_projection.1.weight', 'student_projection.1.bias', 'log_student_temperature'})
sys.path.insert(0, str(AB / 'code'))
spec = importlib.util.spec_from_file_location('cassava_parent_for_OT', AB / 'code/legacy_parent.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
sys.path.insert(0, str(parent.CORE))
sys.path.insert(0, str(ROOT / 'code'))
from pair_model import PairITMModel, pair_usa_loss
from albef_ssl.model import get_tokenizer
from ot_loss import soft_ot_targets, ot_weight

DEVICE = parent.DEVICE
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
sha256, atomic_json, atomic_torch, now = parent.sha256, parent.atomic_json, parent.atomic_torch, parent.now


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'Immutable registration differs: {path}')
    else:
        atomic_json(path, value)


def tensor_digest(state):
    return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                 for _, v in sorted(state.items()))).hexdigest()


def fingerprint():
    paths = [ROOT / 'code' / name for name in ('train_queue.py', 'ot_loss.py', 'prepare_u_pairs.py')]
    paths += [ROOT / 'configs/plan.json', REGISTRATION, ROOT / 'data/manifest.json',
              ROOT / 'audit/u_data_checks.json', ROOT / 'audit/parent_binding.json']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def prepare():
    """Validate immutable inputs and produce ten configs; never constructs a model."""
    plan, manifest = read(ROOT / 'configs/plan.json'), read(ROOT / 'data/manifest.json')
    if (plan['budget_order'] != ['005', '020', '001', '010', '030']
            or plan['variants'] != ['G3', 'G4'] or plan['additional_steps'] != 1600
            or plan['unlabeled_batch'] != 32 or plan['labeled_positive_anchors'] != 16
            or plan['seed'] != SEED or plan['test_evaluation'] or plan['extra_BCE_continuation']):
        raise RuntimeError('Plan differs from the authorized main protocol')
    gate = read(ROOT / 'audit/u_data_checks.json')
    if not gate.get('passed') or gate['manifest_sha256'] != sha256(ROOT / 'data/manifest.json'):
        raise RuntimeError('Prepared U data gate not bound to current manifest')
    if manifest['preparation_code_sha256'] != sha256(ROOT / 'code/prepare_u_pairs.py'):
        raise RuntimeError('U preparation source changed')
    for path, expected in manifest['source_sha256'].items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f'U source identity changed: {path}')
    pipeline = read(AB / 'outputs/pipeline_state.json')
    c_pipeline = read(C_ROOT / 'outputs/pipeline_state.json')
    if pipeline['state'] != 'completed_validation' or c_pipeline['state'] != 'completed_validation':
        raise RuntimeError('Original A/B/C queues must be completed')
    source = parent.source_fingerprint(read(AB / 'data/manifest.json'))
    if source != pipeline['source_fingerprint']:
        raise RuntimeError('Historical A/B code, input or tokenizer fingerprint changed')
    for rel, expected in c_pipeline['branch_fingerprint'].items():
        if sha256(C_ROOT / rel) != expected:
            raise RuntimeError(f'Historical C branch changed: {rel}')
    model_cfg = parent.model_config()
    if parent.md5(Path(model_cfg['checkpoint'])) != model_cfg['checkpoint_md5_expected']:
        raise RuntimeError('Original ALBEF checkpoint differs')
    binding = {'ab_root': str(AB), 'c_root': str(C_ROOT), 'source_fingerprint': source,
               'c_branch_fingerprint': c_pipeline['branch_fingerprint'], 'runs': {}}
    configs = []
    for budget in plan['budget_order']:
        entry = manifest['budgets'][budget]
        for path, expected in ((ROOT / entry['path'], entry['sha256']),
                               (Path(entry['l_pairs_path']), entry['l_pairs_sha256'])):
            if sha256(path) != expected:
                raise RuntimeError(f'Registered pair table changed: {path}')
        a_id = f'apple_itm_random_A_bce_l{budget}_s{SEED}'
        b_id = f'apple_itm_random_B_bce_pairusa_l{budget}_s{SEED}'
        c_id = f'apple_itm_random_C_from_A_best_pairusa_l{budget}_s{SEED}'
        a_dir, b_dir, c_dir = AB / 'outputs' / a_id, AB / 'outputs' / b_id, C_ROOT / 'outputs' / c_id
        a, b, c = (read(p / 'result.json') for p in (a_dir, b_dir, c_dir))
        for result in (a, b, c):
            if result['state'] != 'completed_validation' or result['provenance']['base']['source'] != source:
                raise RuntimeError(f'Uncompleted or changed parent result: {result["run_id"]}')
        for result in (a, b):
            if result['provenance']['base']['run'] != read(AB / 'configs' / (result['run_id'] + '.json')):
                raise RuntimeError('Parent result/config mismatch')
        c_cfg = read(C_ROOT / 'configs' / (c_id + '.json'))
        if c['provenance']['base']['run'] != c_cfg:
            raise RuntimeError('Historical C result/config mismatch')
        a_ckpt = torch.load(a_dir / 'best.pt', map_location='cpu', weights_only=False)
        if (a_ckpt['step'] != a['best_step'] or a_ckpt['validation'] != a['best_validation']
                or a_ckpt['provenance'] != a['provenance'] or len(a_ckpt['trainable_state']) != 34
                or not all(torch.isfinite(v).all() for v in a_ckpt['trainable_state'].values())):
            raise RuntimeError('A best is not the complete registered best checkpoint')
        teacher_dir = AB / 'outputs' / f'teacher_l{budget}_s{SEED}'
        teacher_files = {str(teacher_dir / name): sha256(teacher_dir / name)
                         for name in ('best.pt', 'positive_targets.pt')}
        if teacher_files != b['provenance']['teacher_artifacts']:
            raise RuntimeError('Original teacher artifact hash mismatch')
        if teacher_files != c['provenance']['teacher_artifacts']:
            raise RuntimeError('C teacher does not equal original B teacher')
        teacher_result = read(teacher_dir / 'result.json')
        if teacher_result['state'] != 'completed' or teacher_result['provenance']['source'] != source:
            raise RuntimeError('Teacher result is not bound to parent data/code')
        warm = {'a_best_path': str(a_dir / 'best.pt'), 'a_best_sha256': sha256(a_dir / 'best.pt'),
                'a_best_step': a['best_step'], 'a_run_id': a_id, 'c_run_id': c_id,
                'a_result_sha256': sha256(a_dir / 'result.json'),
                'c_result_sha256': sha256(c_dir / 'result.json'),
                'c_initialization_audit': str(c_dir / 'initialization_audit.json'),
                'c_initialization_sha256': sha256(c_dir / 'initialization_audit.json'),
                'teacher_artifacts': teacher_files,
                'teacher_result_sha256': sha256(teacher_dir / 'result.json')}
        if c_cfg['warmstart']['a_best_sha256'] != warm['a_best_sha256']:
            raise RuntimeError('C and new branches do not share exactly the same A best')
        binding['runs'][budget] = warm
        original = b['provenance']['base']['run']
        for variant in plan['variants']:
            suffix = 'ot' if variant == 'G3' else 'usa_ot'
            cfg = {'run_id': f'apple_itm_{variant}_from_A_best_{suffix}_l{budget}_s{SEED}',
                   'variant': variant, 'budget': budget, 'seed': SEED,
                   'model': copy.deepcopy(original['model']), 'training': copy.deepcopy(original['training']),
                   'image_root': manifest['image_root'], 'pair_file': entry['l_pairs_path'],
                   'validation_file': str(AB / 'data/pairs/validation.csv'),
                   'u_file': str(ROOT / entry['path']), 'u_sha256': entry['sha256'],
                   'u_n_images': entry['n_images'], 'u_n_pairs': entry['n_pairs'],
                   'warmstart': warm, 'ot': plan['ot'], 'u_strong': plan['u_strong'],
                   'u_weak': plan['u_weak'], 'unlabeled_batch': 32,
                   'max_cpu_prefix_cache_gib': plan['max_cpu_prefix_cache_gib'],
                   'caption_protocol': plan['caption_protocol'], 'test_evaluation': False}
            cfg['model']['use_pairusa'] = variant == 'G4'
            cfg['model']['fusion_chunk_size'] = plan['initial_physical_batch']
            configs.append(cfg)
    immutable_json(ROOT / 'audit/parent_binding.json', binding)
    for cfg in configs:
        immutable_json(ROOT / 'configs' / (cfg['run_id'] + '.json'), cfg)
    immutable_json(ROOT / 'configs/queue.json', {'run_ids': [c['run_id'] for c in configs],
                   'manifest_sha256': sha256(ROOT / 'data/manifest.json'), 'total_new_runs': 10})
    return manifest, configs, fingerprint()


def load_u_rows(path, expected_hash=None):
    """Only the five approved public fields; never opens audit_private."""
    path = Path(path)
    if expected_hash and sha256(path) != expected_hash:
        raise RuntimeError('U table fingerprint changed')
    fields = ['pair_id', 'image_id', 'image_path', 'text', 'text_sha256']
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise RuntimeError('Unlabeled loader rejects labels, donor or other metadata')
        rows = list(reader)
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate U pair identifiers')
    for row in rows:
        p = Path(row['image_path'])
        if p.is_absolute() or '..' in p.parts or p.parts[0] != 'images':
            raise RuntimeError('U image must use registered neutral relative path')
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise RuntimeError('U verbatim caption hash mismatch')
    return rows


class Runtime:
    def __init__(self, cfg, eager=False):
        if DEVICE.type != 'cuda':
            raise RuntimeError('This registered queue requires CUDA')
        torch.set_num_threads(8)
        self.cfg = cfg
        self.train_rows = parent.read_pairs(Path(cfg['pair_file']))
        self.val_rows = parent.read_pairs(Path(cfg['validation_file']))
        self.u_rows = load_u_rows(cfg['u_file'], cfg['u_sha256'])
        if (len(self.train_rows) != parent.EXPECTED_COUNTS[cfg['budget']]
                or len(self.val_rows) != 400 or len(self.u_rows) != cfg['u_n_pairs']):
            raise RuntimeError('Wrong L/U/validation counts')
        train_ids = {r['image_id'] for r in self.train_rows}
        u_ids = {r['image_id'] for r in self.u_rows}
        pool_ids = parent.source_ids(parent.V2_DATA / 'train_pool_ids.csv')
        if (train_ids & u_ids or train_ids | u_ids != pool_ids
                or len(u_ids) != cfg['u_n_images']
                or (train_ids | u_ids) & {r['image_id'] for r in self.val_rows}):
            raise RuntimeError('Invalid L/U boundary')
        self.train_index = {r['image_id']: i for i, r in enumerate(self.train_rows)}
        self.model = PairITMModel(cfg['model'])
        spec = cfg['warmstart']
        if sha256(Path(spec['parent_best_path'])) != spec['parent_best_sha256']:
            raise RuntimeError('Parent best changed before initialization')
        parent_best = torch.load(spec['parent_best_path'], map_location='cpu', weights_only=False)
        initial, a_state = self.model.trainable_state(), parent_best['trainable_state']
        expected_extra = EXTRA_KEYS if cfg['variant'] == 'G4' else frozenset()
        if (len(a_state) != 34 or set(initial) - set(a_state) != expected_extra
                or set(a_state) - set(initial)):
            raise RuntimeError('Parent warmstart keys differ from registered 34/39 layout')
        fresh = {k: initial[k].clone() for k in expected_extra}
        self.model.load_trainable_state({**initial, **a_state})
        loaded = self.model.trainable_state()
        if not all(torch.equal(loaded[k], v) for k, v in a_state.items()):
            raise RuntimeError('Parent best transfer is not exact')
        self.initproof = {'exact_copy': True, 'copied_keys': 34, 'trainable_keys': len(loaded),
                          'parent_kind': spec['parent_kind'], 'parent_run_id': spec['parent_run_id'],
                          'parent_best_path': spec['parent_best_path'], 'parent_best_sha256': spec['parent_best_sha256'],
                          'parent_best_step': parent_best['step'], 'parent_trainable_sha256': tensor_digest(a_state),
                          'loaded_shared_sha256': tensor_digest({k: loaded[k] for k in a_state}),
                          'fresh_keys': sorted(expected_extra),
                          'fresh_usa_sha256': tensor_digest(fresh) if fresh else None}
        if fresh and self.initproof['fresh_usa_sha256'] != read(spec['c_initialization_audit'])['fresh_usa_sha256']:
            raise RuntimeError('G4 USA initialization differs from original C')
        self.teacher = None
        if cfg['variant'] == 'G4':
            target_path = AB / 'outputs' / f"teacher_l{cfg['budget']}_s{SEED}" / 'positive_targets.pt'
            if sha256(target_path) != spec['teacher_artifacts'][str(target_path)]:
                raise RuntimeError('Teacher target hash changed')
            target = torch.load(target_path, map_location='cpu', weights_only=False)
            if (target['image_ids'] != [r['image_id'] for r in self.train_rows]
                    or target['view_ids'] != ['canonical', 'horizontal_flip']
                    or tuple(target['positive_vectors'].shape) != (len(self.train_rows), 2, 256)):
                raise RuntimeError('Teacher target identity/order/view mismatch')
            self.teacher = target['positive_vectors'].float().to(DEVICE).detach()
        self.model.to(DEVICE)
        self.opt = parent.optimizer_for(self.model, cfg)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
        self.tokenizer = get_tokenizer(cfg['model']['tokenizer_path'])
        self.cache = parent.PairCache(self.model, self.tokenizer, Path(cfg['image_root']))
        self.physical_batch = cfg['model']['fusion_chunk_size']
        if eager:
            self.cache.prime_images(self.train_rows + self.val_rows, flips=True)
            self.cache.prime_text([t for r in self.train_rows + self.val_rows
                                   for t in (r['positive_text'], r['negative_text'])])


def _text_batch(cache, rows):
    features = [cache.text[r['text_sha256']] for r in rows]
    length = max(f.shape[0] for f in features)
    text = torch.zeros((len(features), length, 768), dtype=torch.float16)
    mask = torch.zeros((len(features), length), dtype=torch.long)
    for i, f in enumerate(features):
        text[i, :len(f)] = f
        mask[i, :len(f)] = 1
    return text, mask


def strong_array(row, step, image_root):
    with Image.open(Path(image_root) / row['image_path']) as image:
        image = image.convert('RGB').resize((384, 384), Image.Resampling.BICUBIC)
        weak = np.asarray(image, dtype=np.uint8).copy()
        rng = random.Random(parent.fixed_seed(SEED, step, 'u-strong/' + row['pair_id']))
        operations = [(TF.adjust_brightness, rng.uniform(0.9, 1.1)),
                      (TF.adjust_contrast, rng.uniform(0.9, 1.1))]
        rng.shuffle(operations)
        for operation, factor in operations:
            image = operation(image, factor)
        strong = np.asarray(image, dtype=np.uint8).copy()
    return strong, float(np.abs(strong.astype(np.float32) - weak.astype(np.float32)).mean())


def _raw_labeled(model, inputs):
    image, pt, pm, nt, nm = inputs
    image = model._image_tail(image)
    positive = model._fuse(image, pt, pm)
    negative = model._fuse(image, nt, nm)
    return torch.cat((positive, negative), dim=0)


def _raw_queries(model, image, text, mask):
    return model._fuse(model._image_tail(image), text, mask)


def _cache_size(cache):
    return sum(x.numel() * x.element_size() for table in (cache.text, cache.images) for x in table.values())


def _accumulate_gradients(rt, inputs, indexes, views, lam, mu, strong_cpu, text_cpu, mask_cpu, q):
    """No optimizer/scaler update here: an OOM can safely retry this whole frame."""
    model, cfg = rt.model, rt.cfg
    with torch.autocast('cuda', dtype=torch.float16):
        logits, student = model.forward_cached(*inputs)
        bce = 0.5 * (F.binary_cross_entropy_with_logits(logits[:, 0].float(), torch.ones(16, device=DEVICE))
                     + F.binary_cross_entropy_with_logits(logits[:, 1].float(), torch.zeros(16, device=DEVICE)))
        usa = torch.zeros((), device=DEVICE)
        if lam:
            usa = pair_usa_loss(rt.teacher[indexes, views], student,
                cfg['training']['pairusa_teacher_temp'], model.student_temperature())
        supervised = bce + lam * usa
    if not torch.isfinite(supervised):
        raise FloatingPointError('Nonfinite supervised loss')
    rt.scaler.scale(supervised).backward()
    bce_value, usa_value = float(bce.detach()), float(usa.detach())
    del logits, student, bce, usa, supervised
    ot_value = 0.0
    if mu:
        for begin in range(0, len(q), rt.physical_batch):
            end = begin + rt.physical_batch
            with torch.autocast('cuda', dtype=torch.float16):
                fused = _raw_queries(model, strong_cpu[begin:end].to(DEVICE),
                    text_cpu[begin:end].to(DEVICE), mask_cpu[begin:end].to(DEVICE))
                u_logits = model.match_head(fused).squeeze(-1)
                piece = F.binary_cross_entropy_with_logits(u_logits.float(), q[begin:end], reduction='sum') / len(q)
            if not torch.isfinite(piece):
                raise FloatingPointError('Nonfinite U soft-label BCE')
            rt.scaler.scale(mu * piece).backward()
            ot_value += float(piece.detach())
            del fused, u_logits, piece
    return bce_value, usa_value, ot_value


def perform_step(rt, step, ot_enabled=True):
    """One successful logical update; all random choices are keyed by the step."""
    started = time.perf_counter()
    cfg, model, cache = rt.cfg, rt.model, rt.cache
    mu = ot_weight(step) if ot_enabled else 0.0
    lam = parent.pairusa_weight(step, cfg['training']) if rt.teacher is not None else 0.0
    selected = parent.choose_unique(rt.train_rows, step)
    cache.prime_images(selected, flips=True)
    inputs = cache.pair_batch(selected, step, augment=True)
    indexes = torch.tensor([rt.train_index[r['image_id']] for r in selected], device=DEVICE)
    views = torch.tensor([parent.fixed_seed(SEED, step, f"flip/{r['image_id']}") & 1 for r in selected], device=DEVICE)
    u_selected, diagnostics, strong_cpu, text_cpu, mask_cpu, q = [], {'active': False}, None, None, None, None
    if mu:
        rng = random.Random(parent.fixed_seed(SEED, step, 'u-pair-sampler'))
        u_selected = rng.sample(rt.u_rows, cfg['unlabeled_batch'])
        unique = {r['image_id']: {'image_id': r['image_id'], 'image_relpath': r['image_path']} for r in u_selected}
        cache.prime_images(list(unique.values()), flips=False)
        cache.prime_text([r['text'] for r in u_selected])
        text_cpu, mask_cpu = _text_batch(cache, u_selected)
        weak_cpu = torch.stack([cache.images[(r['image_id'], 0)] for r in u_selected])
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            anchors = _raw_labeled(model, inputs)
            query_parts = []
            for begin in range(0, len(u_selected), rt.physical_batch):
                end = begin + rt.physical_batch
                query_parts.append(_raw_queries(model, weak_cpu[begin:end].to(DEVICE),
                    text_cpu[begin:end].to(DEVICE), mask_cpu[begin:end].to(DEVICE)))
            queries = torch.cat(query_parts)
        labels = torch.cat((torch.ones(16, device=DEVICE), torch.zeros(16, device=DEVICE)))
        solver_args = {key: cfg['ot'][key] for key in ('epsilon', 'max_iterations', 'tolerance')}
        q, diagnostics = soft_ot_targets(anchors, labels, queries, **solver_args)
        diagnostics['active'] = True
        del anchors, queries, query_parts, weak_cpu
        arrays, deltas = zip(*(strong_array(r, step, cfg['image_root']) for r in u_selected))
        diagnostics['strong_weak_pixel_mae'] = float(np.mean(deltas))
        diagnostics['strong_view_digest'] = hashlib.sha256(np.stack(arrays).tobytes()).hexdigest()
        prefixes = []
        for begin in range(0, len(arrays), rt.physical_batch):
            pixels = torch.from_numpy(np.stack(arrays[begin:begin + rt.physical_batch])).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
            pixels = (pixels - parent.MEAN.to(DEVICE)) / parent.STD.to(DEVICE)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                prefixes.append(model.image_prefix(pixels).detach().cpu().half())
        strong_cpu = torch.cat(prefixes)
        del pixels, prefixes, arrays
    if step % 50 == 0 and _cache_size(cache) > cfg['max_cpu_prefix_cache_gib'] * 1024 ** 3:
        raise MemoryError('CPU prefix cache exceeded registered memory guard')
    factor = parent.schedule(step, cfg['training']['warmup_steps'], cfg['training']['max_steps'])
    for group in rt.opt.param_groups:
        group['lr'] = group['initial_lr'] * factor
    params = [p for p in model.parameters() if p.requires_grad]
    retries = 0
    while retries < 10:
        model.train()
        rt.opt.zero_grad(set_to_none=True)
        try:
            bce_value, usa_value, ot_value = _accumulate_gradients(
                rt, inputs, indexes, views, lam, mu, strong_cpu, text_cpu, mask_cpu, q)
        except torch.cuda.OutOfMemoryError:
            if rt.physical_batch <= 4:
                raise
            rt.opt.zero_grad(set_to_none=True)
            previous_batch = rt.physical_batch
            rt.physical_batch = model.fusion_chunk_size = max(4, previous_batch // 2)
            print(f'[{now()}] physical microbatch {previous_batch} -> {rt.physical_batch} after OOM, logical batch unchanged', flush=True)
            # Leave the exception frame before collecting its graph tensors.
        else:
            # These calls are deliberately outside the OOM retry handler. An
            # error during an optimizer update must restore durable last.pt.
            rt.scaler.unscale_(rt.opt)
            norm = torch.nn.utils.clip_grad_norm_(params, cfg['training']['max_grad_norm'], error_if_nonfinite=False)
            old_scale = rt.scaler.get_scale()
            rt.scaler.step(rt.opt)
            rt.scaler.update()
            if bool(torch.isfinite(norm)):
                break
            if rt.scaler.get_scale() >= old_scale:
                raise FloatingPointError('Nonfinite gradient without scale backoff')
            retries += 1
        gc.collect()
        torch.cuda.empty_cache()
    else:
        raise FloatingPointError(f'FP16 scale retries exhausted at step {step}')
    return {'step': step, 'itm_bce': bce_value, 'pairusa': usa_value, 'ot': ot_value,
            'lambda': lam, 'mu': mu, 'weighted_usa': lam * usa_value, 'weighted_ot': mu * ot_value,
            'loss': bce_value + lam * usa_value + mu * ot_value, 'grad_norm': float(norm),
            'loss_scale': rt.scaler.get_scale(), 'overflow_retries': retries,
            'physical_batch': rt.physical_batch, 'diagnostics': diagnostics,
            'l_image_ids': [r['image_id'] for r in selected],
            'u_pair_ids': [r['pair_id'] for r in u_selected], 'u_image_ids': [r['image_id'] for r in u_selected],
            'step_seconds': time.perf_counter() - started, 'utc': now()}


def capture_training_state(rt):
    return {'trainable_state': rt.model.trainable_state(), 'optimizer': copy.deepcopy(rt.opt.state_dict()),
            'scaler': copy.deepcopy(rt.scaler.state_dict()), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(), 'physical_batch': rt.physical_batch}


def validate_resume_checkpoint(checkpoint, provenance, expected_keys):
    """Fail closed on incomplete/nonfinite states, even if metadata matches."""
    required = {'trainable_state', 'optimizer', 'scaler', 'torch_rng', 'cuda_rng',
                'python_rng', 'numpy_rng', 'physical_batch', 'provenance', 'step',
                'best_snapshot', 'best_scores', 'seen_u_pairs', 'seen_u_images', 'active_seconds'}
    if not required <= set(checkpoint) or checkpoint['provenance'] != provenance:
        raise RuntimeError('Resume checkpoint is incomplete or provenance differs')
    step = checkpoint['step']
    if type(step) is not int or not 50 <= step <= 1600 or step % 50:
        raise RuntimeError('Resume checkpoint has an invalid durable step')
    if set(checkpoint['trainable_state']) != set(expected_keys) or checkpoint['physical_batch'] not in (4, 8, 16):
        raise RuntimeError('Resume trainable keys or physical batch differ')

    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(v) for v in value)
        if isinstance(value, float):
            return bool(np.isfinite(value))
        return True

    if not finite_tree(checkpoint) or checkpoint['scaler'].get('scale', 0) <= 0:
        raise RuntimeError('Resume checkpoint contains nonfinite values or invalid scaler')
    if not checkpoint['optimizer'].get('state'):
        raise RuntimeError('Resume optimizer moments are missing')
    best, scores = checkpoint['best_snapshot'], checkpoint['best_scores']
    if step >= 100:
        if (not isinstance(best, dict) or type(best.get('step')) is not int
                or not 100 <= best['step'] <= step or best['step'] % 100
                or set(best.get('trainable_state', {})) != set(expected_keys)
                or not isinstance(scores, torch.Tensor) or tuple(scores.shape) != (400, 2)
                or not isinstance(best.get('validation'), dict)):
            raise RuntimeError('Resume best snapshot/scores do not belong to this durable step')
    elif best is not None or scores is not None:
        raise RuntimeError('Resume has a best snapshot before the first validation')


def restore_training_state(rt, checkpoint):
    rt.model.load_trainable_state(checkpoint['trainable_state'])
    rt.opt.load_state_dict(checkpoint['optimizer'])
    rt.scaler.load_state_dict(checkpoint['scaler'])
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    random.setstate(checkpoint['python_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    rt.physical_batch = rt.model.fusion_chunk_size = checkpoint['physical_batch']


@contextmanager
def queue_lock():
    import msvcrt
    with (ROOT / '.queue.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b' ')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            with parent.process_lock():
                yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def reconcile_logs(out, durable_step):
    for name in ('train.jsonl', 'validation.jsonl'):
        path = out / name
        if not path.exists():
            continue
        lines = path.read_text(encoding='utf-8').splitlines()
        retained = []
        dirty = False
        for line in lines:
            try:
                item = json.loads(line)
                if int(item['step']) <= durable_step:
                    retained.append(line)
                else:
                    dirty = True
            except (ValueError, KeyError, TypeError):
                dirty = True
        if dirty:
            backup = out / 'recovery_backups' / f'{time.time_ns()}_{name}'
            backup.parent.mkdir(exist_ok=True)
            shutil.copy2(path, backup)
            parent.atomic_bytes(path, ('\n'.join(retained) + ('\n' if retained else '')).encode('utf-8'))


def write_best(out, provenance, best, scores, initproof):
    atomic_torch(out / 'best.pt', {**best, 'provenance': provenance, 'initialization': initproof})
    rows = parent.read_pairs(Path(provenance['run']['validation_file']))
    parent.atomic_bytes(out / 'best_validation_predictions.csv',
        parent.validation_prediction_csv(rows, scores, best['validation']['validation_selected_threshold']))


def run_one(cfg, fp):
    out = OUTPUTS / cfg['run_id']
    out.mkdir(parents=True, exist_ok=True)
    provenance = {'source': fp, 'run': cfg, 'test_evaluation': False}
    if (out / 'result.json').exists():
        result = read(out / 'result.json')
        if result['provenance'] != provenance:
            raise RuntimeError('Completed branch provenance changed')
        return result
    atomic_json(out / 'status.json', {'state': 'initializing_model', 'step': 0, 'target_steps': 1600,
        'provenance': provenance, 'updated_utc': now(), 'pid': os.getpid()})
    rt = Runtime(cfg, eager=False)
    immutable_json(out / 'initialization_audit.json', rt.initproof)
    start, best, best_scores, seen_pairs, seen_images, seconds = 0, None, None, set(), set(), 0.0
    if (out / 'last.pt').exists():
        checkpoint = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        validate_resume_checkpoint(checkpoint, provenance, rt.model.trainable_state())
        restore_training_state(rt, checkpoint)
        start, best, best_scores = checkpoint['step'], checkpoint['best_snapshot'], checkpoint['best_scores']
        seen_pairs, seen_images = set(checkpoint['seen_u_pairs']), set(checkpoint['seen_u_images'])
        seconds = checkpoint['active_seconds']
        reconcile_logs(out, start)
        if best is not None:
            for name in ('best.pt', 'best_validation_predictions.csv'):
                p = out / name
                if p.exists():
                    backup = out / 'recovery_backups' / f'{time.time_ns()}_{name}'
                    backup.parent.mkdir(exist_ok=True)
                    shutil.copy2(p, backup)
            write_best(out, provenance, best, best_scores.numpy(), rt.initproof)
        del checkpoint
        print(f'[{now()}] {cfg["run_id"]}: restored complete step {start}', flush=True)
    atomic_json(out / 'status.json', {'state': 'caching_frozen_prefixes', 'step': start,
        'target_steps': 1600, 'provenance': provenance, 'updated_utc': now(), 'pid': os.getpid()})
    cache_started = time.perf_counter()
    rt.cache.prime_images(rt.train_rows + rt.val_rows, flips=True)
    rt.cache.prime_text([t for r in rt.train_rows + rt.val_rows for t in (r['positive_text'], r['negative_text'])])
    seconds += time.perf_counter() - cache_started
    print(f'[{now()}] {cfg["run_id"]}: cache ready, start={start}', flush=True)
    best_key = (best['validation']['paired_accuracy'], best['validation']['auroc']) if best else (-float('inf'), -float('inf'))
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start + 1, 1601):
            tick = time.perf_counter()
            record = perform_step(rt, step)
            seen_pairs.update(record.pop('u_pair_ids'))
            seen_images.update(record.pop('u_image_ids'))
            record['u_unique_pairs_seen'] = len(seen_pairs)
            record['u_unique_images_seen'] = len(seen_images)
            parent.append_jsonl(out / 'train.jsonl', record)
            if step % 100 == 0:
                metrics, scores = parent.validation(rt.model, rt.cache, rt.val_rows)
                parent.append_jsonl(out / 'validation.jsonl', {'step': step, **metrics, 'utc': now()})
                key = (metrics['paired_accuracy'], metrics['auroc'])
                if key > best_key:
                    best_key = key
                    best = {'step': step, 'validation': metrics, 'trainable_state': rt.model.trainable_state()}
                    best_scores = torch.from_numpy(scores.copy())
                    write_best(out, provenance, best, scores, rt.initproof)
                print(f'[{now()}] {cfg["run_id"]}: step {step}/1600 paired={metrics["paired_accuracy"]:.5f} auc={metrics["auroc"]:.5f}', flush=True)
            seconds += time.perf_counter() - tick
            if step % 50 == 0:
                atomic_torch(out / 'last.pt', {**capture_training_state(rt), 'provenance': provenance,
                    'step': step, 'best_step': best['step'] if best else None,
                    'best_snapshot': best, 'best_scores': best_scores,
                    'seen_u_pairs': sorted(seen_pairs), 'seen_u_images': sorted(seen_images),
                    'active_seconds': seconds})
            if step % 10 == 0:
                atomic_json(out / 'status.json', {'state': 'running', 'step': step,
                    'target_steps': 1600, 'best_step': best['step'] if best else None,
                    'last_durable_step': step // 50 * 50, 'updated_utc': now(), 'pid': os.getpid(),
                    'step_seconds': record['step_seconds'], 'mu': record['mu'], 'lambda': record['lambda'],
                    'u_unique_images_seen': len(seen_images), 'physical_batch': rt.physical_batch,
                    'active_seconds': seconds, 'provenance': provenance})
        result = {'run_id': cfg['run_id'], 'state': 'completed_validation', 'best_step': best['step'],
            'best_validation': best['validation'], 'final_step': 1600,
            'parent_kind': cfg['warmstart']['parent_kind'],
            'parent_best_step': cfg['warmstart']['parent_best_step'],
            'provenance': provenance, 'completed_utc': now(), 'active_seconds': seconds,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'u_unique_pairs_seen': len(seen_pairs),
            'u_unique_images_seen': len(seen_images), 'u_draws': (1600 - 100) * 32,
            'initialization': rt.initproof, 'independent_test_evaluation': False}
        atomic_json(out / 'result.json', result)
        atomic_json(out / 'status.json', {**result, 'step': 1600, 'target_steps': 1600, 'updated_utc': now()})
        return result
    except BaseException as exc:
        atomic_json(out / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'ot_diagnostics': getattr(exc, 'diagnostics', None), 'updated_utc': now(),
            'attempted_step': locals().get('step'), 'provenance': provenance})
        atomic_json(out / 'status.json', {'state': 'failed', 'step': locals().get('step', start) - 1,
            'error': repr(exc), 'updated_utc': now(), 'provenance': provenance})
        raise
    finally:
        del rt
        gc.collect()
        torch.cuda.empty_cache()


def report():
    script = ROOT / 'code/report_progress.py'
    if script.exists():
        subprocess.run([sys.executable, '-X', 'utf8', str(script)], cwd=ROOT, check=True)


def main():
    raise RuntimeError('Cassava OT stages must be started through code/cassava_queue.py')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report()
        return
    _, configs, fp = prepare()
    if args.prepare:
        print(json.dumps({'prepared_runs': len(configs), 'fingerprint': fp}, ensure_ascii=False), flush=True)
        return
    verification = read(ROOT / 'audit/training_verification.json')
    if not verification.get('passed') or verification.get('fingerprint') != fp:
        raise RuntimeError('Real-data initialization/update/resume checks must pass for these exact sources')
    numerical = read(ROOT / 'audit/ot_numerical_checks.json')
    if (not numerical.get('passed')
            or numerical.get('source_sha256', {}).get('ot_loss.py') != fp['code/ot_loss.py']):
        raise RuntimeError('OT numerical verification is not passed')
    OUTPUTS.mkdir(exist_ok=True)
    with queue_lock(), parent.prevent_system_sleep():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
        state_path = OUTPUTS / 'pipeline_state.json'
        state = {'state': 'starting', 'active_run': None, 'run_ids': [c['run_id'] for c in configs],
                 'fingerprint': fp, 'pid': os.getpid(), 'test_evaluation': False, 'started_utc': now()}
        if state_path.exists():
            old = read(state_path)
            if old['fingerprint'] != fp or old['run_ids'] != state['run_ids']:
                raise RuntimeError('Pipeline fingerprint differs; refusing overwrite')
            state['started_utc'] = old['started_utc']
        atomic_json(state_path, state)
        try:
            for cfg in configs:
                state.update(state='running', active_run=cfg['run_id'], updated_utc=now())
                atomic_json(state_path, state)
                run_one(cfg, fp)
                report()
            state.update(state='completed_validation', active_run=None, completed_utc=now(), updated_utc=now())
            atomic_json(state_path, state)
            report()
        except BaseException as exc:
            state.update(state='failed', error=repr(exc), updated_utc=now())
            atomic_json(state_path, state)
            raise


if __name__ == '__main__':
    main()
