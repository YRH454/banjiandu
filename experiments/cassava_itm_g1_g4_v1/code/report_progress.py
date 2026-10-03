# -*- coding: utf-8 -*-
"""Read-only experiment inspection and offline HTML/JSON reporting.

Never imports the trainer, loads a model, uses CUDA, or evaluates the test set.
On Windows, shared-delete reads avoid blocking atomic status replacement.
"""
from __future__ import annotations

import csv
import ctypes
import hashlib
import html
import io
import json
import math
import shutil
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NAME = "木薯实验结果_最新汇总"
LOCAL_TZ = timezone(timedelta(hours=8))
METHODS = {"S1": "BCE", "S2": "BCE + Pair-USA", "S3": "BCE + OT", "S4": "BCE + Pair-USA + OT"}
CLASSES = ("CBB", "CBSD", "CGM", "CMD", "Healthy")
STATE_NAMES = {"completed_validation": "已完成", "running": "训练中", "caching_frozen_prefixes": "缓存准备",
               "initializing_model": "模型初始化", "preparing_teacher": "教师准备", "failed": "失败", "pending": "待运行"}
FILE_HASHES = {}


def shared_bytes(path: Path) -> bytes:
    if __import__("os").name != "nt":
        data = path.read_bytes()
    else:
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileSizeEx.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong))
        kernel.GetFileSizeEx.restype = wintypes.BOOL
        kernel.ReadFile.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
        kernel.ReadFile.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        # READ | WRITE | DELETE sharing; never creates, edits or locks a source file.
        handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 7, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            if error in (2, 3):
                raise FileNotFoundError(path)
            raise ctypes.WinError(error)
        try:
            size = ctypes.c_longlong()
            if not kernel.GetFileSizeEx(handle, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            chunks, remaining = [], size.value
            buffer = ctypes.create_string_buffer(65536)
            while remaining:
                read = wintypes.DWORD()
                if not kernel.ReadFile(handle, buffer, min(len(buffer), remaining), ctypes.byref(read), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not read.value:
                    break
                chunks.append(buffer.raw[:read.value])
                remaining -= read.value
            data = b"".join(chunks)
        finally:
            kernel.CloseHandle(handle)
    FILE_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def read_json(path, optional=False):
    try:
        return json.loads(shared_bytes(path).decode("utf-8-sig"))
    except FileNotFoundError:
        if optional:
            return None
        raise


def read_jsonl(path):
    try:
        data = shared_bytes(path)
    except FileNotFoundError:
        return []
    lines = data.decode("utf-8-sig").splitlines()
    if data and not data.endswith(b"\n"):
        lines = lines[:-1]  # A live writer may be halfway through its last record.
    return [json.loads(line) for line in lines if line.strip()]


def read_csv(path):
    return list(csv.DictReader(io.StringIO(shared_bytes(path).decode("utf-8-sig"))))


def local_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(LOCAL_TZ).strftime("%m-%d %H:%M:%S") if value else "—"


def threshold_metrics(scores, threshold):
    decisions = scores >= threshold
    tp, fp = int(decisions[:, 0].sum()), int(decisions[:, 1].sum())
    fn, tn = len(scores) - tp, len(scores) - fp
    truth = np.tile(np.array([1, 0]), (len(scores), 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall, specificity = tp / len(scores), tn / len(scores)
    return {"threshold": float(threshold), "accuracy": (tp + tn) / (2 * len(scores)),
            "balanced_accuracy": (recall + specificity) / 2, "precision": precision,
            "recall": recall, "specificity": specificity, "negative_accept_rate": fp / len(scores),
            "f1": float(f1_score(truth.ravel(), decisions.ravel(), zero_division=0)),
            "macro_f1": float(f1_score(truth.ravel(), decisions.ravel(), average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(truth.ravel(), decisions.ravel())),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def score_metrics(scores, threshold):
    truth = np.tile(np.array([1, 0]), (len(scores), 1))
    probability = 1.0 / (1.0 + np.exp(-scores))
    selected = threshold_metrics(scores, threshold)
    return {"paired_accuracy": float(np.mean(np.where(scores[:, 0] > scores[:, 1], 1.0,
                                                       np.where(scores[:, 0] == scores[:, 1], 0.5, 0.0)))),
            "auroc": float(roc_auc_score(truth.ravel(), scores.ravel())),
            "average_precision": float(average_precision_score(truth.ravel(), scores.ravel())),
            "validation_selected_threshold": float(threshold),
            "balanced_accuracy_at_validation_threshold": selected["balanced_accuracy"],
            "f1_at_validation_threshold": selected["f1"],
            "positive_recall_at_validation_threshold": selected["recall"],
            "false_caption_accept_rate_at_validation_threshold": selected["negative_accept_rate"],
            "mean_positive_probability": float(probability[:, 0].mean()),
            "mean_negative_probability": float(probability[:, 1].mean()),
            "mean_margin": float((scores[:, 0] - scores[:, 1]).mean()), "n_anchors": len(scores)}


def verify_predictions(result, cfg, rows, reference, expected_source):
    if result["state"] != "completed_validation" or result["run_id"] != cfg["run_id"]:
        raise ValueError("Result identity/state mismatch")
    provenance = result["provenance"]
    base = provenance.get("base", provenance)
    if base["run"] != cfg or base["source"] != expected_source:
        raise ValueError("Result config/source provenance mismatch")
    keys = ("image_id", "negative_source_image_id", "source_text_sha256", "negative_text_sha256")
    if len(rows) != 400 or [[r[k] for k in keys] for r in rows] != [[r[k] for k in keys] for r in reference]:
        raise ValueError("Fixed 400-anchor validation pairs differ")
    scores = np.array([[float(row["positive_logit"]), float(row["negative_logit"])] for row in rows], dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite saved prediction")
    threshold = result["best_validation"]["validation_selected_threshold"]
    checked = score_metrics(scores, threshold)
    for key, stored in result["best_validation"].items():
        if key in checked and not math.isclose(checked[key], stored, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Saved-prediction metric mismatch: {key}")
    for row, score in zip(rows, scores):
        if (not math.isclose(float(row["validation_threshold"]), threshold, rel_tol=1e-6, abs_tol=1e-6)
                or int(row["positive_accepted"]) != int(score[0] >= threshold)
                or int(row["negative_accepted"]) != int(score[1] >= threshold)):
            raise ValueError("Saved decisions/threshold mismatch")
    strata = []
    for code in CLASSES:
        subset = scores[[i for i, row in enumerate(rows) if row["class_code"] == code]]
        if len(subset):
            strata.append({"class_code": code, "n_anchors": len(subset), **score_metrics(subset, threshold)})
    return {"passed": True, "n_anchors": len(rows), "n_pairs": 2 * len(rows),
            "source": "best_validation_predictions.csv; no model inference", "strata": strata,
            "selected_threshold": threshold_metrics(scores, threshold), "fixed_logit_zero": threshold_metrics(scores, 0.0)}


def telemetry():
    result = {"processes": [], "gpu": None}
    try:
        import psutil
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                command = " ".join(process.info["cmdline"] or [])
                if "python" in (process.info["name"] or "").lower() and "--run" in command and any(
                        name in command for name in ("cassava_queue.py", "cassava_fast_queue.py", "recover_status_lock_handoff.py")):
                    if Path(process.cwd()).resolve() == ROOT.resolve():
                        result["processes"].append({"pid": process.pid, "command": command})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        result["processes_available"] = False
    executable = shutil.which("nvidia-smi")
    if not executable:
        candidates = sorted(Path("C:/Windows/System32/DriverStore/FileRepository").glob("nv*/nvidia-smi.exe"))
        executable = str(candidates[0]) if candidates else None
    if executable:
        try:
            output = subprocess.check_output([executable, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                               "--format=csv,noheader,nounits"], timeout=8, text=True)
            name, utilization, used, total = next(csv.reader(io.StringIO(output)))
            result["gpu"] = {"name": name.strip(), "utilization_percent": float(utilization),
                             "memory_used_mib": float(used), "memory_total_mib": float(total), "kind": "point_in_time_whole_gpu"}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return result


def collect():
    plan = read_json(ROOT / "configs/plan.json")
    split = read_json(ROOT / "data/splits/split_manifest.json")
    source = read_json(ROOT / "audit/source_audit.json")
    manifest = read_json(ROOT / "data/manifest.json")
    original = read_json(ROOT / "outputs/pipeline_state.json")
    fast = read_json(ROOT / "outputs/fast_pipeline_state.json", optional=True)
    handoff = read_json(ROOT / "audit/acceleration_handoff.json", optional=True)
    pipeline = fast or original
    reference = read_csv(ROOT / manifest["files"]["validation"])
    runs = []
    for budget in plan["budget_order"]:
        for stage in (("S1", "S2") if budget == "100" else ("S1", "S2", "S3", "S4")):
            version = "original" if budget == "005" else "fast"
            run_id = f"cassava_itm_{'' if version == 'original' else 'fast_'}{stage.lower()}_l{budget}_s{plan['training_seed']}"
            directory = ROOT / "outputs" / run_id
            cfg = read_json(ROOT / "configs" / f"{run_id}.json", optional=True)
            status = read_json(directory / "status.json", optional=True) or {"state": "pending"}
            result = read_json(directory / "result.json", optional=True)
            training = read_jsonl(directory / "train.jsonl")
            validations = read_jsonl(directory / "validation.jsonl")
            latest = training[-1] if training else {}
            step = max(int(status.get("step") or 0), int(latest.get("step") or 0))
            completed = bool(result and result.get("state") == "completed_validation")
            record = {"run_id": run_id, "stage": stage, "method": METHODS[stage], "budget": budget,
                      "budget_percent": int(budget), "l_images": manifest["source_counts"][budget],
                      "u_images": split["summary"]["train"]["count"] - manifest["source_counts"][budget],
                      "version": version, "state": "completed_validation" if completed else status.get("state", "pending"),
                      "completed": completed, "step": 1600 if completed else step, "target_steps": 1600,
                      "last_reported_durable_step": status.get("last_durable_step", (int(status.get("step") or 0) // 50) * 50),
                      "best_step": result["best_step"] if completed else status.get("best_step"),
                      "result": result, "validations": validations, "latest_train": latest,
                      "config_file": f"../configs/{run_id}.json", "result_file": f"../outputs/{run_id}/result.json",
                      "prediction_file": f"../outputs/{run_id}/best_validation_predictions.csv",
                      "ot_tolerance": cfg["ot"]["tolerance"] if cfg and "ot" in cfg else None,
                      "fusion_microbatch": cfg["model"]["fusion_chunk_size"] if cfg else (8 if version == "original" else 16),
                      "wall_train_log_seconds": None, "recent_timing": None}
            if len(training) > 1:
                record["wall_train_log_seconds"] = (datetime.fromisoformat(training[-1]["utc"]) -
                                                       datetime.fromisoformat(training[0]["utc"])).total_seconds()
            timed = [row for row in training[-50:] if isinstance(row.get("step_seconds"), (int, float)) and row["step_seconds"] > 0]
            if timed:
                record["recent_timing"] = {"n": len(timed), "from_step": timed[0]["step"], "to_step": timed[-1]["step"],
                                            "median_step_seconds": statistics.median(row["step_seconds"] for row in timed),
                                            "nonconverged_ot": sum(bool(row.get("diagnostics", {}).get("active")) and not row["diagnostics"]["converged"] for row in timed),
                                            "overflow_retries": sum(row.get("overflow_retries", 0) for row in timed),
                                            "kind": "descriptive_window_not_acceleration_verdict"}
            if completed:
                if status.get("state") != "completed_validation" or status.get("step") != 1600 or not cfg:
                    raise ValueError(f"Completed result/status/config disagree: {run_id}")
                if len(training) != 1600 or [row["step"] for row in training] != list(range(1, 1601)):
                    raise ValueError(f"Completed training log is not 1600 contiguous successful steps: {run_id}")
                record["prediction_check"] = verify_predictions(result, cfg, read_csv(directory / "best_validation_predictions.csv"),
                                                                reference, original["source_fingerprint"] if version == "original" else fast["source_fingerprint"])
                warm = cfg.get("warmstart") or cfg["model"].get("warmstart") or {}
                record["parent_kind"] = warm.get("parent_kind", "ALBEF-4M")
                record["parent_run_id"] = warm.get("parent_run_id")
                record["parent_best_step"] = warm.get("parent_best_step")
                record["final_step"] = 1600
            runs.append(record)
    by_id = {run["run_id"]: run for run in runs}
    def selected_path(run, visited=None):
        visited = set() if visited is None else visited
        if run["run_id"] in visited:
            raise ValueError("Cyclic warmstart lineage")
        visited.add(run["run_id"])
        parent = by_id.get(run.get("parent_run_id"))
        return (selected_path(parent, visited) if parent else 0) + run["best_step"]
    for run in runs:
        if run["completed"]:
            run["selected_path_updates"] = selected_path(run)
    completed = [run for run in runs if run["completed"]]
    active = by_id.get(pipeline.get("active_run"))
    script_sha256 = hashlib.sha256(shared_bytes(Path(__file__))).hexdigest()
    snapshot = {"generated_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"), "experiment_root": str(ROOT),
                "scope": "stored_validation_results_only_no_test_or_new_inference", "plan": plan,
                "source_summary": {key: source[key] for key in ("source_repo", "source_revision", "class_counts", "non_leaf_count", "source_integrity_passed")},
                "split_summary": split["summary"], "validation_pairs": 800,
                "validation_class_counts": {code: sum(row["class_code"] == code for row in reference) for code in CLASSES},
                "pipeline_state": pipeline.get("state"), "active_run_id": pipeline.get("active_run"),
                "pipeline_error": pipeline.get("error"), "handoff_state": handoff.get("state") if handoff else None,
                "runs": runs, "completed_stages": len(completed), "planned_stages": len(runs),
                "successful_logged_updates": len(completed) * 1600 + (active["step"] if active and not active["completed"] else 0),
                "planned_updates": len(runs) * 1600, "telemetry": telemetry(), "prediction_checks_passed": True,
                "report_script_sha256": script_sha256, "inspection_file_hashes": dict(FILE_HASHES)}
    return snapshot


def h(value):
    return html.escape(str(value), quote=True)


def number(value):
    return f"{value:.4f}" if value is not None else "—"


def table(headers, rows, css=""):
    head = "".join(f"<th>{h(label)}</th>" for label in headers)
    body = "".join(f"<tr{attrs}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>" for attrs, cells in rows)
    return f'<div class="table-wrap {css}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(snapshot):
    runs = snapshot["runs"]
    completed = [run for run in runs if run["completed"]]
    active = next((run for run in runs if run["run_id"] == snapshot["active_run_id"]), None)
    best = max(completed, key=lambda run: (run["result"]["best_validation"]["paired_accuracy"], run["result"]["best_validation"]["auroc"])) if completed else None
    gpu = snapshot["telemetry"]["gpu"]
    progress = snapshot["successful_logged_updates"] / snapshot["planned_updates"] * 100
    badges = lambda run: '<span class="badge">原版 / 8</span>' if run["version"] == "original" else '<span class="badge fast">加速版 / 16</span>'
    cards = [("已完成阶段", f'{len(completed)} / {len(runs)}', "不含教师与工程 smoke"),
             ("当前训练", f'{active["budget_percent"]}% {active["stage"]}' if active else "—", f'{active["step"]} / 1600 成功日志步' if active else snapshot["pipeline_state"]),
             ("最高已完成配对准确率", number(best["result"]["best_validation"]["paired_accuracy"]) if best else "—", f'{best["budget_percent"]}% {best["stage"]} · 验证选模' if best else "—"),
             ("固定验证对", "400 + 400", "400正对、400随机错配负对"),
             ("训练种子", str(snapshot["plan"]["training_seed"]), "单种子，无均值±标准差"),
             ("GPU利用率快照", f'{gpu["utilization_percent"]:.0f}%' if gpu else "未读取", f'{gpu["memory_used_mib"] / 1024:.2f} / {gpu["memory_total_mib"] / 1024:.0f} GiB' if gpu else "并非平均利用率")]
    cards_html = "".join(f'<div class="card"><div class="label">{h(label)}</div><div class="value">{h(value)}</div><div class="muted">{h(note)}</div></div>' for label, value, note in cards)
    filters = '<button class="filter on" data-filter="all" aria-pressed="true">全部</button>' + "".join(
        f'<button class="filter" data-filter="{budget}" aria-pressed="false">{int(budget)}%</button>' for budget in snapshot["plan"]["budget_order"])
    primary, threshold, fixed, timings = [], [], [], []
    for run in completed:
        metric = run["result"]["best_validation"]
        check = run["prediction_check"]
        attrs = f' data-budget="{run["budget"]}"'
        label = f'<a href="#{h(run["run_id"])}">{run["budget_percent"]}% · {run["stage"]}</a>'
        parent = "ALBEF-4M" if run["parent_kind"] == "ALBEF-4M" else f'{h(run["parent_kind"])} best @{run["parent_best_step"]}'
        primary.append((attrs, [label, h(run["method"]), badges(run), parent, str(run["best_step"]), str(run["selected_path_updates"]),
                                *[number(metric[key]) for key in ("paired_accuracy", "auroc", "average_precision", "f1_at_validation_threshold")]]))
        selected = check["selected_threshold"]
        threshold.append((attrs, [label, *[number(selected[key]) for key in ("threshold", "precision", "recall", "specificity", "negative_accept_rate", "balanced_accuracy", "f1", "macro_f1", "mcc")],
                                  f'{selected["tp"]} / {selected["fp"]} / {selected["tn"]} / {selected["fn"]}']))
        zero = check["fixed_logit_zero"]
        fixed.append((attrs, [label, *[number(zero[key]) for key in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "negative_accept_rate")],
                              *[number(metric[key]) for key in ("mean_positive_probability", "mean_negative_probability", "mean_margin")]]))
        timing = run["recent_timing"]
        timings.append((attrs, [label, str(run["l_images"]), str(run["u_images"]),
                                number(run["result"].get("active_seconds", 0) / 60) if run["result"].get("active_seconds") is not None else "未记录",
                                number(run["wall_train_log_seconds"] / 60) if run["wall_train_log_seconds"] is not None else "—",
                                str(run["result"].get("u_unique_images_seen", "N/A")), str(run["result"].get("u_unique_pairs_seen", "N/A")),
                                number(run["result"]["peak_gpu_bytes"] / 1024**3) if run["result"].get("peak_gpu_bytes") is not None else "未记录",
                                number(timing["median_step_seconds"]) if timing else "未记录",
                                str(run["ot_tolerance"]) if run["ot_tolerance"] is not None else "N/A"]))
    matrix = []
    for budget in snapshot["plan"]["budget_order"]:
        cells = [f'{int(budget)}%', str(next(run["l_images"] for run in runs if run["budget"] == budget))]
        for stage in METHODS:
            match = next((run for run in runs if run["budget"] == budget and run["stage"] == stage), None)
            if match is None:
                cells.append('<span class="muted">N/A · 无U</span>')
            elif match["completed"]:
                cells.append(f'<span class="done">已完成</span><small>best @{match["best_step"]} / final 1600</small>')
            elif match["state"] != "pending":
                cells.append(f'<span class="live">{h(STATE_NAMES.get(match["state"], match["state"]))}</span><small>{match["step"]} / 1600</small>')
            else:
                cells.append('<span class="muted">待运行</span>')
        matrix.append(("", cells))
    lineage = table(["阶段", "损失", "初始化来源", "新增成功步", "注意"], [
        ("", ["S1", "图文匹配 BCE", "ALBEF-4M", "1600", "不是五类病害分类 BCE"]),
        ("", ["S2", "BCE + 自定义 Pair-USA", "同预算 S1 best", "1600", "不复制 S3 或独立初始化B"]),
        ("", ["S3", "BCE + OT", "同预算 S1 best", "1600", "100% 无U，故不运行"]),
        ("", ["S4", "BCE + Pair-USA + OT", "同预算 S3 best", "1600", "34个公共键继承，5个USA键新初始化"])])
    insights = []
    for budget in snapshot["plan"]["budget_order"]:
        group = {run["stage"]: run for run in completed if run["budget"] == budget}
        if "S1" in group:
            baseline = group["S1"]["result"]["best_validation"]["paired_accuracy"]
            gains = [f'{stage}: {(run["result"]["best_validation"]["paired_accuracy"] - baseline) * 100:+.2f} 个百分点'
                     for stage, run in group.items() if stage != "S1"]
            if gains:
                insights.append(f'<li>{int(budget)}% 相对 S1 的配对排序准确率变化：{h("；".join(gains))}。仅为当前验证集描述，不能排除新增训练阶段的作用。</li>')
    details = []
    for run in completed:
        metric = run["result"]["best_validation"]
        strata_rows = [("", [h(row["class_code"]), str(row["n_anchors"]), *[number(row[key]) for key in
                        ("paired_accuracy", "auroc", "average_precision", "positive_recall_at_validation_threshold", "false_caption_accept_rate_at_validation_threshold", "mean_margin")]])
                       for row in run["prediction_check"]["strata"]]
        history_rows = [(f' class="best-row"' if row["step"] == run["best_step"] else "", [str(row["step"]),
                         *[number(row[key]) for key in ("paired_accuracy", "auroc", "average_precision", "f1_at_validation_threshold")],
                         "best" if row["step"] == run["best_step"] else ""]) for row in run["validations"]]
        details.append(f'<details class="run-detail" id="{h(run["run_id"])}" data-budget="{run["budget"]}"><summary>{run["budget_percent"]}% {run["stage"]} · {h(run["method"])} · best @{run["best_step"]}</summary><div class="detail-body">'
                       f'<p class="muted">{h(run["run_id"])} · 完成 {local_time(run["result"]["completed_utc"])} · 本阶段最终1600步 · 选中权重路径累计 {run["selected_path_updates"]} 步</p>'
                       f'<p class="sources"><a href="{h(run["config_file"])}">配置 JSON</a><a href="{h(run["result_file"])}">原始 result.json</a><a href="{h(run["prediction_file"])}">best 预测 CSV</a></p>'
                       '<h3>病例类别分层诊断</h3><p class="muted">按锚图来源类别分组，沿用全体400图选定阈值；不是五类分类指标，也没有在各类别上重新调阈值。</p>'
                       + table(["来源类别", "锚图数", "配对Acc", "AUROC", "AP", "正对Recall", "错配接受率↓", "平均logit差"], strata_rows)
                       + '<h3>验证历史</h3><p class="muted">best按配对Acc优先、AUROC次优的字典序选取；best之后仍训练至1600步，不是提前停止。</p>'
                       + table(["阶段步数", "配对Acc", "AUROC", "AP", "F1（验证阈值）", "选模"], history_rows) + '</div></details>')
    active_html = '<p>当前无活动学生阶段。</p>'
    if active:
        timing = active["recent_timing"]
        last_val = active["validations"][-1] if active["validations"] else None
        active_html = f'<p><strong>{active["budget_percent"]}% {active["stage"]} · {h(active["method"])}</strong> · {h(STATE_NAMES.get(active["state"], active["state"]))}</p><p>成功日志步数 <strong>{active["step"]}/1600</strong>；最近状态报告的落盘步数 {active["last_reported_durable_step"]}。训练进度和报告均为生成时快照。</p>'
        if last_val:
            active_html += f'<p>最近一次验证 @第{last_val["step"]}步：配对Acc {number(last_val["paired_accuracy"])} · AUROC {number(last_val["auroc"])} · AP {number(last_val["average_precision"])}。<strong>本组尚未完成，不计入正式结果表。</strong></p>'
        if timing:
            active_html += f'<p class="muted">当前计时窗口 {timing["from_step"]}–{timing["to_step"]}（{timing["n"]}步）：step_seconds 中位数 {number(timing["median_step_seconds"])}秒；OT未收敛记录 {timing["nonconverged_ot"]}，溢出重试 {timing["overflow_retries"]}。该统计不构成严格加速对照，U缓存状态、预算和文本分布均可能不同。</p>'
        active_html += f'<p class="muted">训练进程：{h(", ".join(str(process["pid"]) for process in snapshot["telemetry"]["processes"]) or "未检出")}；GPU快照：{h(gpu["name"]) if gpu else "未读取"}。</p>'
    split_rows = [("", [h(code), str(snapshot["source_summary"]["class_counts"][code]),
                        *[str(snapshot["split_summary"][name]["class_counts"][code]) for name in ("train", "validation", "test")],
                        str(snapshot["validation_class_counts"][code])]) for code in CLASSES]
    counts = {name: snapshot["split_summary"][name]["count"] for name in ("train", "validation", "test")}
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>木薯图文匹配实验 · 最新结果汇总</title>
<style>
:root{{--ink:#182b35;--muted:#64747d;--line:#dce4e5;--accent:#087d72;--paper:#fff;--bg:#f3f6f5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1450px;margin:auto;padding:36px 28px 64px}}header{{margin-bottom:24px}}.eyebrow{{color:var(--accent);font-size:12px;font-weight:750;letter-spacing:2px}}h1{{font-size:31px;line-height:1.3;margin:7px 0 10px}}h2{{font-size:20px;margin:0 0 12px}}h3{{font-size:16px;margin:22px 0 9px}}p{{margin:9px 0}}a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}.muted,.label{{color:var(--muted);font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card,section{{background:var(--paper);border:1px solid var(--line);border-radius:13px}}.card{{padding:19px 22px}}.value{{font-size:28px;font-weight:730;font-variant-numeric:tabular-nums;margin:5px 0}}section{{padding:23px;margin-top:20px}}nav{{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}}.notice{{background:#fff8e7;border:1px solid #eddca8;border-radius:9px;padding:14px 17px;margin-top:18px;font-size:13px}}.bar{{height:8px;background:#e0eae6;border-radius:8px;overflow:hidden;margin:13px 0 5px}}.bar span{{display:block;background:var(--accent);height:100%}}.filters{{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0 16px}}button{{cursor:pointer;font:inherit}}.filter{{border:1px solid var(--line);border-radius:18px;background:white;padding:4px 13px;font-size:13px;color:var(--muted)}}.filter.on{{background:var(--accent);color:white;border-color:var(--accent)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid #e9edee}}th{{background:#edf3f0;color:#42585e;font-weight:650}}td:first-child,th:first-child{{text-align:left}}tr:last-child td{{border-bottom:0}}tbody tr:hover{{background:#f5f9f7}}.badge{{display:inline-block;background:#edf2fb;color:#456087;padding:2px 7px;border-radius:5px;font-size:11px}}.badge.fast{{background:#e5f5ee;color:#177754}}small{{display:block;color:var(--muted);font-size:11px}}.done{{color:#147e5a;font-weight:650}}.live{{color:#ae731c;font-weight:650}}.best-row{{background:#eaf6ee}}details{{border:1px solid var(--line);border-radius:9px;margin:11px 0;background:white}}summary{{padding:15px 18px;cursor:pointer;font-weight:600}}.detail-body{{padding:0 18px 21px}}.sources{{display:flex;gap:18px;font-size:13px}}.note-list{{font-size:13px;padding-left:22px}}.note-list li{{margin:7px 0}}[hidden]{{display:none!important}}footer{{margin-top:25px;font-size:12px;color:var(--muted)}}@media(max-width:760px){{main{{padding:22px 13px}}h1{{font-size:25px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:16px}}.value{{font-size:23px}}}}@media(max-width:450px){{.cards{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}.filters{{display:none}}section{{border-radius:0;break-inside:avoid}}.table-wrap{{overflow:visible}}table{{font-size:10px}}th,td{{padding:6px 5px}}}}
</style></head><body><main>
<header><div class="eyebrow">CASSAVA / IMAGE–TEXT MATCHING / SINGLE SEED</div><h1>木薯 S1–S4 实验结果汇总</h1><p class="muted">快照生成：{h(snapshot['generated_at'])} · 全部数值来自本地训练记录与已保存验证预测 · 离线HTML，不自动刷新</p><nav><a href="#progress">流水进度</a><a href="#results">已完成结果</a><a href="#threshold">阈值与混淆矩阵</a><a href="#diagnostics">逐组诊断</a><a href="#protocol">协议与限制</a></nav></header>
<div class="cards">{cards_html}</div><div class="notice"><strong>任务口径：</strong>这是图文匹配二分类和同图正/负caption排序，不是五类病害分类。全部结果来自同一固定400图验证子集，未进行独立Test评估。随机异病例caption负例可能语义相容，标签不是人工语义金标准。</div>
<section id="progress"><h2>流水进度</h2><div class="bar"><span style="width:{progress:.2f}%"></span></div><p class="muted">成功日志步数 {snapshot['successful_logged_updates']:,}/{snapshot['planned_updates']:,}（{progress:.2f}%）；这是更新步数比例，不是剩余耗时比例。</p>{table(['标注预算','L图数','S1 · BCE','S2 · +USA','S3 · +OT','S4 · +USA+OT'],matrix)}<h3>当前运行快照</h3>{active_html}</section>
<section id="results"><h2>已完成组 · 最佳验证权重</h2><p class="muted">所有已完成组均训练至本阶段1600步。表中best步数是验证选模点，不是停止点；路径累计步数按继承父best计算，不是总计算成本。所有指标以0–1显示，保留四位小数。配对Acc比较同图正caption与随机错配caption的logit，平局计0.5；AUROC/AP对800个图文对统一计算，AP指average precision。</p><div class="filters">{filters}</div>{table(['预算/组','损失','版本/融合微批','父权重','阶段best步','best路径累计步','配对Acc↑','AUROC↑','AP↑','F1↑（验证阈值）'],primary)}<h3>当前可描述的差异</h3><ul class="note-list">{''.join(insights)}</ul><p class="muted">未完成组不填入正式表，也不使用当前临时验证值冒充最终结果。不同预算与执行版本不可直接作为严格加速或等算力消融对照。</p></section>
<section id="threshold"><h2>验证选定阈值 · 二分类完整指标</h2><p class="muted">threshold是logit阈值，不是概率。训练器在同一验证集上选择最大正类F1的阈值；下表沿用该阈值，不重新优化。混淆矩阵顺序TP/FP/TN/FN，其中“正”指来源匹配图文对。</p>{table(['预算/组','logit阈值','Precision↑','正对Recall↑','Specificity↑','错配接受率↓','BalancedAcc↑','正类F1↑','Macro-F1↑','MCC↑','TP / FP / TN / FN'],threshold)}<h3>统一固定阈值 · logit=0 / probability=0.5</h3><p class="muted">下表由同一best预测CSV补算，不训练新模型，不改变原选模；用于区分验证调阈值收益与固定0.5判别表现。</p>{table(['预算/组','Accuracy↑','BalancedAcc↑','Precision↑','正对Recall↑','Specificity↑','正类F1↑','错配接受率↓','正对平均概率','负对平均概率','平均logit差'],fixed)}</section>
<section><h2>样本访问与耗时口径</h2>{table(['预算/组','L图数','U池图数','已记录active时间/min','训练日志窗口/min','实际U图访问数','实际U对访问数','torch峰值/GiB','末50步中位数/s','OT容差'],timings)}<p class="muted">active时间仅在原记录存在时展示，通常不含进程等待/加载等全部开销；训练日志窗口由首尾成功步UTC相减，不含前置教师/缓存准备，可能包含暂停。两者不是统一端到端耗时。末50步中位数为描述性统计，缓存状态可能不同，不代表专项复核已证明提速。GPU快照是整卡读数，torch峰值是模型分配口径。S1/S2未记录step_seconds与torch峰值，故显示“未记录”，不伪造。</p></section>
<section id="diagnostics"><h2>逐组类别诊断、验证历史与来源</h2><p class="muted">{len(completed)}组已完成结果的预测重算检查状态：<strong class="done">全部通过</strong>。每组400锚图/800图文对；图文对ID、caption哈希、logit、阈值、接受决策及原始指标已交叉核对。</p>{''.join(details)}</section>
<section id="protocol"><h2>实验定义与可复现边界</h2>{lineage}<h3>数据与划分</h3><p>共 {sum(snapshot['source_summary']['class_counts'].values()):,} 图，五类；Train {counts['train']:,} / Validation {counts['validation']:,} / Test {counts['test']:,}，固定80/10/10分层划分。训练种子 {snapshot['plan']['training_seed']}，划分种子 {snapshot['plan']['split_seed']}。根/茎/其他非叶片主体标记 {snapshot['source_summary']['non_leaf_count']} 张保留，whole-plant另保留。</p>{table(['来源类别','全体','Train','Validation全池','Test封存','固定验证锚图'],split_rows)}<ul class="note-list"><li>学生与教师使用完整blind caption，384-token上限覆盖已审计实测最长360；guided和疾病类别不作为训练输入。</li><li>L/U均位于Train边界内。低标注预算中U允许caption、隐藏匹配构造标签；100%没有U，S3/S4为N/A，因此总共22组，不是24组。</li><li>负例为固定随机异病例完整原始blind caption，状态random_other_case_unverified；可以出现语义相容假负例，不能宣称人工审核金标准。</li><li>逻辑L16正锚形成32对；OT每步U32，完整32×32关系；Pair-USA完整16×16关系。各阶段1600成功更新，新阶段独立优化器和学习率日程。</li><li>5%四组保留原融合微批8与OT容差1e-5。后续18组使用登记的微批16与OT容差1e-4，epsilon=0.1、最多100次保持；执行版本必须标注，四位小数显示不是精度误差保证。</li><li>S1只有基础阶段，S2/S3增加一个阶段，S4在S3 best上再增加阶段；未设置纯BCE等时长续训控制组。S4不能视为从S1直接启动的严格单因素对照。</li><li>单种子、验证选模与验证调阈值存在选择偏差。没有独立Test泛化结果、置信区间或多种子稳定性结论。分层表只是匹配诊断，不是疾病分类性能。</li><li>历史5% S4状态文件占用故障已从完整断点恢复并完成；已完成结果保留，原管线标记handed_off_after_5pct。报告生成只读源文件，不运行GPU核验、训练或新评估。</li></ul><p class="muted">来源：{h(snapshot['source_summary']['source_repo'])} · revision <code>{h(snapshot['source_summary']['source_revision'])}</code> · <a href="{NAME}.json">本次完整JSON快照与核验哈希</a></p></section>
<footer>报告脚本：code/report_progress.py · 仅写reports目录 · 本机来源链接需要保留原实验目录结构。所有表格由已登记记录生成，无外部字体、脚本或联网请求。</footer>
</main><script>document.querySelectorAll('.filter').forEach(button=>button.addEventListener('click',()=>{{const budget=button.dataset.filter;document.querySelectorAll('.filter').forEach(b=>{{b.classList.toggle('on',b===button);b.setAttribute('aria-pressed',String(b===button))}});document.querySelectorAll('tr[data-budget],details.run-detail[data-budget]').forEach(item=>item.hidden=budget!=='all'&&item.dataset.budget!==budget)}}));</script></body></html>'''


def main():
    snapshot = collect()
    content = render(snapshot)
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path, html_path = (REPORTS / f"{NAME}.{suffix}" for suffix in ("json", "html"))
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(content, encoding="utf-8")
    print(json.dumps({"html": str(html_path), "json": str(json_path), "generated_at": snapshot["generated_at"],
                      "completed_stages": snapshot["completed_stages"], "planned_stages": snapshot["planned_stages"],
                      "active_run_id": snapshot["active_run_id"], "prediction_checks_passed": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
