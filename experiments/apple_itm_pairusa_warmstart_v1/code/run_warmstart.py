"""Isolated A-best -> Pair-USA branch; never edits the registered parent queue."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'apple_itm_pairusa_random_v2'
OUTPUTS = ROOT / 'outputs'
EXTRA_KEYS = frozenset({'student_projection.0.weight', 'student_projection.0.bias',
                        'student_projection.1.weight', 'student_projection.1.bias',
                        'log_student_temperature'})
REGISTRATION = ROOT / 'reports/执行登记_C_A最佳权重续训_20260929.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.atomic-', suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def immutable_json(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f'Registered contents changed: {path}')
    else:
        atomic_json(path, value)


def snapshot_digest(path):
    result = hashlib.sha256()
    for child in sorted(Path(path).rglob('*')):
        if child.is_file():
            result.update(str(child.relative_to(path)).replace('\\', '/').encode())
            result.update(digest(child).encode('ascii'))
    return result.hexdigest()


def verify_parent_source():
    binding = read(ROOT / 'audit/parent_binding.json')
    pipeline = read(PARENT / 'outputs/pipeline_state.json')
    if pipeline['source_fingerprint'] != binding['source_fingerprint']:
        raise RuntimeError('Parent pipeline source binding changed')
    for name, expected in binding['source_fingerprint'].items():
        if name == 'tokenizer_snapshot':
            config = read(PARENT / 'configs/apple_itm_random_A_bce_l005_s20260825.json')
            actual = snapshot_digest(Path(config['model']['tokenizer_path']))
        else:
            path = Path(name)
            actual = digest(path if path.is_absolute() else PARENT / path)
        if actual != expected:
            raise RuntimeError(f'Parent fingerprint mismatch: {name}')
    return binding['source_fingerprint']


def branch_fingerprint():
    return {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in (
        Path(__file__), ROOT / 'configs/plan.json', REGISTRATION, ROOT / 'audit/parent_binding.json')}


def prepare():
    plan = read(ROOT / 'configs/plan.json')
    if Path(plan['parent_root']).resolve() != PARENT.resolve() or plan['additional_steps'] != 1600:
        raise RuntimeError('Unexpected parent root or phase length')
    source = read(PARENT / 'outputs/pipeline_state.json')['source_fingerprint']
    immutable_json(ROOT / 'audit/parent_binding.json', {'parent_root': str(PARENT), 'source_fingerprint': source})
    verify_parent_source()
    return plan


def load_backend():
    sys.path.insert(0, str(PARENT / 'code'))
    spec = importlib.util.spec_from_file_location('registered_parent_queue_for_C', PARENT / 'code/train_queue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(module.CORE) not in sys.path:
        sys.path.insert(0, str(module.CORE))
    return module


def tensor_digest(state):
    return hashlib.sha256(b''.join(value.detach().cpu().contiguous().numpy().tobytes()
                                  for _, value in sorted(state.items()))).hexdigest()


def merge_a_into_student(model, a_state):
    """Strictly transfer all A parameters, retaining exactly five new USA keys."""
    import torch
    initial = model.trainable_state()
    if len(a_state) != 34 or set(initial) - set(a_state) != EXTRA_KEYS or set(a_state) - set(initial):
        raise RuntimeError('A-to-C parameter keys differ from registered 34+5 topology')
    for name, value in a_state.items():
        if value.shape != initial[name].shape or not torch.isfinite(value).all():
            raise RuntimeError(f'Invalid A parameter: {name}')
    fresh = {key: initial[key].clone() for key in EXTRA_KEYS}
    model.load_trainable_state({**initial, **a_state})
    loaded = model.trainable_state()
    if any(not torch.equal(loaded[key], value) for key, value in a_state.items()):
        raise RuntimeError('A weights were not transferred exactly')
    if any(not torch.equal(loaded[key], value) for key, value in fresh.items()):
        raise RuntimeError('Fresh USA initialization was unexpectedly changed')
    return {'copied_keys': len(a_state), 'fresh_keys': sorted(EXTRA_KEYS),
            'a_trainable_sha256': tensor_digest(a_state),
            'loaded_shared_sha256': tensor_digest({key: loaded[key] for key in a_state}),
            'fresh_usa_sha256': tensor_digest(fresh), 'exact_copy': True}


def make_config(budget, source):
    a_id = f'apple_itm_random_A_bce_l{budget}_s20260825'
    b_id = f'apple_itm_random_B_bce_pairusa_l{budget}_s20260825'
    a_dir, b_dir = PARENT / 'outputs' / a_id, PARENT / 'outputs' / b_id
    a_result, b_result = read(a_dir / 'result.json'), read(b_dir / 'result.json')
    for run_id, result in ((a_id, a_result), (b_id, b_result)):
        if result['state'] != 'completed_validation':
            raise RuntimeError(f'Parent student incomplete: {run_id}')
        if result['provenance']['base']['source'] != source:
            raise RuntimeError(f'Parent student source mismatch: {run_id}')
        if result['provenance']['base']['run'] != read(PARENT / 'configs' / f'{run_id}.json'):
            raise RuntimeError(f'Parent student config mismatch: {run_id}')
    cfg = copy.deepcopy(b_result['provenance']['base']['run'])
    if a_result['provenance']['base']['run']['model'] != cfg['model']:
        raise RuntimeError('A/B base-model settings differ')
    teacher_artifacts = b_result['provenance']['teacher_artifacts']
    teacher_dir = PARENT / 'outputs' / f'teacher_l{budget}_s20260825'
    expected_teacher_paths = {str(teacher_dir / name) for name in ('best.pt', 'positive_targets.pt')}
    if set(teacher_artifacts) != expected_teacher_paths:
        raise RuntimeError('B teacher paths differ from corresponding budget')
    for path, expected in teacher_artifacts.items():
        if digest(path) != expected:
            raise RuntimeError(f'B teacher artifact changed: {path}')
    teacher_result = read(teacher_dir / 'result.json')
    teacher_prov = teacher_result['provenance']
    if (teacher_result['state'] != 'completed' or teacher_prov['source'] != source
            or teacher_prov['budget'] != budget or teacher_prov['model'] != cfg['model']
            or teacher_prov['train_pair_file_sha256'] != digest(PARENT / cfg['pair_file'])
            or teacher_prov['validation_pair_file_sha256'] != digest(PARENT / cfg['validation_file'])):
        raise RuntimeError('Teacher provenance mismatch')
    cfg['run_id'] = f'apple_itm_random_C_from_A_best_pairusa_l{budget}_s20260825'
    # Original run_student enables its unchanged USA path for methods starting with B.
    cfg['method'] = 'B_bce_pairusa_warmstart_A_best'
    cfg['variant'] = 'C_A_best_then_BCE_PairUSA'
    cfg['warmstart'] = {
        'branch_fingerprint': branch_fingerprint(), 'a_run_id': a_id, 'b_run_id': b_id,
        'a_best_path': str(a_dir / 'best.pt'), 'a_best_sha256': digest(a_dir / 'best.pt'),
        'a_result_sha256': digest(a_dir / 'result.json'), 'a_best_step': a_result['best_step'],
        'b_result_sha256': digest(b_dir / 'result.json'),
        'teacher_result_sha256': digest(teacher_dir / 'result.json'),
        'teacher_artifacts': teacher_artifacts,
        'additional_steps': 1600, 'optimizer_state': 'fresh_not_available_in_A_best',
        'sampling_step': 'restart_phase_step_1', 'source_artifacts_read_only': True}
    cfg['model']['warmstart'] = {key: cfg['warmstart'][key] for key in
                                ('a_best_path', 'a_best_sha256', 'a_best_step', 'a_run_id')}
    cfg['model']['warmstart']['audit_path'] = str(OUTPUTS / cfg['run_id'] / 'initialization_audit.json')
    if cfg['training']['max_steps'] != 1600:
        raise RuntimeError('Parent B is not the registered 1600-step experiment')
    return cfg


def install_warmstart(backend):
    import torch
    import pair_model
    original_model = pair_model.PairITMModel

    class ABestWarmStartedModel(original_model):
        def __init__(self, config):
            super().__init__(config)
            spec = config['warmstart']
            if digest(spec['a_best_path']) != spec['a_best_sha256']:
                raise RuntimeError('A best.pt changed before model initialization')
            checkpoint = torch.load(spec['a_best_path'], map_location='cpu', weights_only=True)
            parent_result = read(Path(spec['a_best_path']).parent / 'result.json')
            if (checkpoint['provenance'] != parent_result['provenance']
                    or checkpoint['step'] != spec['a_best_step']
                    or checkpoint['validation'] != parent_result['best_validation']
                    or parent_result['run_id'] != spec['a_run_id']):
                raise RuntimeError('A best.pt does not match its registered best result')
            proof = merge_a_into_student(self, checkpoint['trainable_state'])
            atomic_json(spec['audit_path'], {**proof, 'a_best_step': checkpoint['step'],
                'a_best_sha256': spec['a_best_sha256'], 'initialization': 'A_best_weights_only',
                'optimizer_and_scaler': 'fresh_for_new_phase', 'updated_utc': now()})

    def existing_teacher(cfg, train_rows, val_rows, tokenizer, source_fp):
        # Rebuild and compare the entire immutable configuration before returning a teacher.
        if make_config(cfg['budget'], source_fp) != cfg:
            raise RuntimeError('Warm-start/teacher configuration changed')
        return PARENT / 'outputs' / f"teacher_l{cfg['budget']}_s20260825" / 'best.pt'

    pair_model.PairITMModel = ABestWarmStartedModel
    backend.train_teacher = existing_teacher
    backend.OUTPUTS = OUTPUTS
    return ABestWarmStartedModel


@contextmanager
def waiter_lock():
    import msvcrt
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.queue.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b' ')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another C continuation waiter/worker is active') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def parent_queue_lock(backend):
    """Completion JSON can precede the original process releasing its lock."""
    while True:
        context = backend.process_lock()
        try:
            context.__enter__()
            break
        except RuntimeError as exc:
            if 'holds the queue lock' not in str(exc):
                raise
            time.sleep(5)
    try:
        yield
    finally:
        context.__exit__(None, None, None)


def report_progress():
    pipeline_path = OUTPUTS / 'pipeline_state.json'
    state = read(pipeline_path) if pipeline_path.exists() else {'state': 'not_started'}
    results = []
    for path in sorted(OUTPUTS.glob('apple_itm_random_C_*/result.json')):
        result = read(path)
        cfg = result['provenance']['base']['run']
        results.append({'run_id': result['run_id'], 'budget': cfg['budget'],
                        'a_best_step': cfg['warmstart']['a_best_step'],
                        'best_phase_step': result['best_step'], 'validation': result['best_validation']})
    atomic_json(ROOT / 'reports/latest_progress.json', {'updated_utc': now(), 'pipeline': state,
        'completed_C': len(results), 'target_C': 6, 'results': results,
        'test_evaluation': False, 'limitation': 'Additional optimization; not a compute-matched USA-only ablation.'})


def run():
    with waiter_lock():
        plan = prepare()
        identity = branch_fingerprint()
        audit = read(ROOT / 'audit/verification.json')
        if not audit.get('passed') or audit['branch_fingerprint'] != identity:
            raise RuntimeError('Current C branch lacks matching initialization/resume verification')
        state_path = OUTPUTS / 'pipeline_state.json'
        state = {'state': 'waiting_for_parent', 'active_run': None, 'branch_fingerprint': identity,
                 'budget_order': plan['budget_order'], 'pid': os.getpid(), 'test_evaluation': False}
        if state_path.exists() and read(state_path)['branch_fingerprint'] != identity:
            raise RuntimeError('Existing C pipeline fingerprint changed; refusing resume')
        while True:
            parent = read(PARENT / 'outputs/pipeline_state.json')
            if parent['state'] == 'completed_validation':
                break
            state.update(state='waiting_for_parent', parent_state=parent['state'],
                         parent_active_run=parent.get('active_run'), updated_utc=now())
            atomic_json(state_path, state)
            time.sleep(30)
        source = verify_parent_source()
        backend = load_backend()
        # This is the original parent's lock, not the separate waiter lock above.
        with parent_queue_lock(backend), backend.prevent_system_sleep():
            if read(PARENT / 'outputs/pipeline_state.json')['state'] != 'completed_validation':
                raise RuntimeError('Parent is no longer complete after acquiring GPU queue lock')
            if branch_fingerprint() != identity:
                raise RuntimeError('C source changed while waiting')
            import torch
            if backend.DEVICE.type != 'cuda':
                raise RuntimeError('CUDA is required for C training')
            torch.set_num_threads(8)
            if backend.md5(Path(backend.model_config()['checkpoint'])) != backend.model_config()['checkpoint_md5_expected']:
                raise RuntimeError('Original ALBEF weight fingerprint changed')
            manifest = backend.load_manifest(require_gate=True)
            configs = [make_config(budget, source) for budget in plan['budget_order']]
            for cfg in configs:
                immutable_json(ROOT / 'configs' / f"{cfg['run_id']}.json", cfg)
            from albef_ssl.model import get_tokenizer
            tokenizer = get_tokenizer(backend.model_config()['tokenizer_path'])
            install_warmstart(backend)
            torch.manual_seed(backend.SEED)
            torch.cuda.manual_seed_all(backend.SEED)
            torch.backends.cudnn.benchmark = True
            state['run_ids'] = [cfg['run_id'] for cfg in configs]
            for cfg in configs:
                verify_parent_source()
                if branch_fingerprint() != identity:
                    raise RuntimeError('C source changed during queue execution')
                state.update(state='running', active_run=cfg['run_id'], updated_utc=now())
                atomic_json(state_path, state)
                train_rows = backend.read_pairs(PARENT / cfg['pair_file'])
                val_rows = backend.read_pairs(PARENT / cfg['validation_file'])
                if len(train_rows) != backend.EXPECTED_COUNTS[cfg['budget']] or len(val_rows) != 400:
                    raise RuntimeError('C data counts differ from registered budgets')
                train_ids = {row['image_id'] for row in train_rows}
                val_ids = {row['image_id'] for row in val_rows}
                if (not train_ids <= backend.source_ids(backend.V2_DATA / 'budgets' / f"train_{cfg['budget']}.csv")
                        or not val_ids <= backend.source_ids(backend.V2_DATA / 'validation.csv') or train_ids & val_ids):
                    raise RuntimeError('C data split boundary mismatch')
                result = backend.run_student(cfg, train_rows, val_rows, tokenizer, source)
                proof = read(OUTPUTS / cfg['run_id'] / 'initialization_audit.json')
                if result['initial_shared_sha256'] != proof['a_trainable_sha256']:
                    raise RuntimeError('C initialization did not equal corresponding A best')
                report_progress()
            state.update(state='completed_validation', active_run=None, completed_utc=now(), updated_utc=now())
            atomic_json(state_path, state)
            report_progress()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print(json.dumps({'state': 'prepared_not_started', 'branch_fingerprint': branch_fingerprint()}, ensure_ascii=False))
    elif args.report:
        report_progress()
    else:
        try:
            run()
        except Exception as exc:
            atomic_json(OUTPUTS / 'pipeline_failure.json', {'utc': now(), 'error': repr(exc), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
