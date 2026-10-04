"""Reuse archived mathematical/model components without importing old queues."""
from __future__ import annotations

import importlib.util
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

from .spec import ROOT


@lru_cache(maxsize=None)
def source_module(relative_path, dependency_path=None):
    path = ROOT / relative_path
    name = "_fair_v3_" + relative_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("losses")
    if dependency_path:
        sys.modules["losses"] = source_module(dependency_path)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        if dependency_path:
            if previous is None:
                sys.modules.pop("losses", None)
            else:
                sys.modules["losses"] = previous
    return module


def numerical_components():
    """No model assets, old profile, source mutation, queue, or CUDA init."""
    return dict(
        losses=source_module("experiments/multicrop_itm_mt_fixmatch_v1/code/losses.py"),
        soft_sim=source_module("experiments/multicrop_itm_soft_simmatch_v1/code/ssl_algorithms.py",
                               "experiments/multicrop_itm_mt_fixmatch_v1/code/losses.py"),
        free=source_module("experiments/multicrop_itm_freematch_v1/code/freematch_algorithm.py"),
        ot=source_module("experiments/banana_itm_s1_s4_v1/code/ot_loss.py"))


def model_components():
    package = sys.modules.get("albef_ssl")
    expected_core = ROOT / "experiments/multicrop_itm_mt_fixmatch_v1/core"
    if package is not None and (not getattr(package, "__file__", None) or not Path(package.__file__).resolve().is_relative_to(expected_core.resolve())):
        raise RuntimeError("Use a fresh fair-v3 worker process; another ALBEF implementation is already imported")
    return source_module("experiments/multicrop_itm_mt_fixmatch_v1/code/pair_model.py")


def reference_paths():
    paths = [
        "experiments/multicrop_itm_mt_fixmatch_v1/code/losses.py",
        "experiments/multicrop_itm_mt_fixmatch_v1/code/pair_model.py",
        "experiments/multicrop_itm_mt_fixmatch_v1/code/metrics.py",
        "experiments/multicrop_itm_mt_fixmatch_v1/code/data_backend.py",
        "experiments/multicrop_itm_soft_simmatch_v1/code/ssl_algorithms.py",
        "experiments/multicrop_itm_freematch_v1/code/freematch_algorithm.py",
        "experiments/banana_itm_s1_s4_v1/code/ot_loss.py"]
    paths.extend(str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted(
        (ROOT / "experiments/multicrop_itm_mt_fixmatch_v1/core").rglob("*")) if p.is_file() and p.suffix in (".py", ".json"))
    return paths


def source_fingerprint():
    """Public source bytes only; not private input/host/GPU admission."""
    from .spec import PROTOCOL_PATH, fingerprint
    paths = set(reference_paths())
    paths.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "src/fair_benchmark").glob("*.py"))
    paths.update((str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"), "tools/fair_benchmark.py"))
    files = [dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest()) for name in sorted(paths)]
    return dict(format="fair_source_fingerprint_v3", source_sha256=fingerprint(files),
                public_code_only_not_execution_admission=True, files=files)
