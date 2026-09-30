"""Read-only verification of v2 registration; writes only a new audit report."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import datetime

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'apple_multimodal_ce_bce_v2'

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def ids(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    values = [row['image_id'] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate image IDs: {path}')
    return set(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT.parents[1] / 'weights/ALBEF_4M.pth')
    args = parser.parse_args()
    registration = json.loads((OLD / 'audit/registration.json').read_text(encoding='utf-8-sig'))
    required = [f'data/budgets/train_{budget}.csv' for budget in ('001','005','010','020','030','100')]
    required += ['data/validation.csv', 'data/test.csv', 'data/train_pool_ids.csv',
                 'data/blind_captions.json', 'audit/excluded_image_ids.csv']
    hashes = {name: digest(OLD / name) for name in required}
    for name, value in hashes.items():
        if value != registration['artifact_sha256'][name]:
            raise ValueError(f'Registered source SHA256 mismatch: {name}')
    train = ids(OLD / 'data/budgets/train_100.csv')
    val = ids(OLD / 'data/validation.csv')
    # Test IDs only: do not load any test captions or use labels.
    test = ids(OLD / 'data/test.csv')
    if train & val or train & test or val & test:
        raise ValueError('Train/validation/test image-ID leakage')
    previous = set()
    counts = {}
    for budget in ('001','005','010','020','030','100'):
        current = ids(OLD / f'data/budgets/train_{budget}.csv')
        if not previous <= current <= train:
            raise ValueError(f'Non-nested budget {budget}')
        counts[budget] = len(current)
        previous = current
    md5 = digest(args.checkpoint, 'md5')
    if md5 != '3c876d776a8e0ce61e2285fc9897f0b3':
        raise ValueError('Original checkpoint MD5 mismatch')
    report = {
        'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'passed_source_identity_only_not_negative_quality',
        'source_root': str(OLD), 'source_registration_sha256': digest(OLD / 'audit/registration.json'),
        'source_artifact_sha256': hashes, 'checkpoint': str(args.checkpoint), 'checkpoint_md5': md5,
        'budget_sizes': counts, 'validation_ids': len(val), 'held_out_test_ids': len(test),
        'test_captions_read': False, 'test_model_evaluation': False,
        'nested_budgets': True, 'splits_disjoint': True,
    }
    (ROOT / 'audit').mkdir(parents=True, exist_ok=True)
    (ROOT / 'audit/source_preflight.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
