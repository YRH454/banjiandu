# -*- coding: utf-8 -*-
"""Read-only source aggregation for G1/G2 and the ten new OT runs.

Does not import training/torch or access hidden U provenance. Only numeric
predictions and validation IDs are retained from the existing CSV files.
All output is derived reporting under the new experiment's reports folder.
"""

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
AB_ROOT = ROOT.parent / "apple_itm_pairusa_random_v2"
C_ROOT = ROOT.parent / "apple_itm_pairusa_warmstart_v1"
BUDGETS = ("001", "005", "010", "020", "030", "100")
L_COUNTS = {"001": 133, "005": 668, "010": 1337, "020": 2674, "030": 4011, "100": 13373}
LABELS = {"G1": "BCE（原 A）", "G2": "A best → BCE + USA（原 C）",
          "G3": "A best → BCE + OT", "G4": "A best → BCE + USA + OT"}
NOTES = [
    "任务为图文匹配二分类；USA 为自定义 Pair-USA。G2 复用 A-best 续训 C，不使用独立初始化的历史 B。",
    "G3/G4 采用新的 caption 可用、匹配标签受限协议：使用训练池 U 的 blind caption，但不读取 U 的隐藏配对来源标签。G1/G2 只使用 L。",
    "G1 为原 BCE 阶段；G2/G3/G4 从对应 A best 各新增 1600 步。G1 少一个训练阶段，收益不能全部归因于辅助损失；未增加纯 BCE 续训对照。",
    "100% 时 U=0，仅展示 G1/G2 两项参考；G3/G4 为 N/A。本轮新增任务总数严格为 10，smoke 不计入。",
    "仅单种子 20260825；固定 400 个验证图像、800 个配对，验证集同时用于选模与阈值选择；未做独立测试集评估。",
    "负例为完整 blind caption 随机异病例固定错排（random_other_case_unverified），可能语义相容；没有人工金标准声明。",
    "paired accuracy 是同图正分数大于负分数的比例，平局记 0.5。原阈值沿用每组验证集选出的最大 F1 阈值；另报告 logit=0 的固定阈值。",
    "Brier 为全部 800 个配对的概率均方误差。ECE10 将正类概率按 [0,1] 等宽分成 10 桶，以各桶平均预测概率与正例频率之差加权。",
    "报告只派生数值，不改写原 result/status/predictions。未记录的耗时、显存或访问数标为空，不能当作 0。",
]


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def safe_div(n, d):
    return float(n / d) if d else 0.0


def classification_metrics(y, scores, threshold):
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "logit_threshold": float(threshold),
        "accuracy": safe_div(tp + tn, len(y)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": safe_div(tp, tp + fp), "recall": recall,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "specificity": specificity, "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def read_numeric_predictions(path):
    # Access only score/ID/threshold columns. Caption and disease columns are
    # neither selected for storage nor copied into any generated artifact.
    ids, scores, thresholds = [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"positive_logit", "negative_logit", "validation_threshold"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing numeric prediction columns: " + str(path))
        for row in reader:
            ids.append(row.get("image_id", ""))
            scores.append([float(row["positive_logit"]), float(row["negative_logit"])])
            thresholds.append(float(row["validation_threshold"]))
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) != 400 or scores.shape != (400, 2) or len(set(ids)) != 400:
        raise ValueError("Expected 400 distinct paired validation images: " + str(path))
    if not np.isfinite(scores).all() or not np.isfinite(thresholds).all():
        raise ValueError("Non-finite validation predictions: " + str(path))
    if len(set(thresholds)) != 1:
        raise ValueError("Inconsistent per-row threshold: " + str(path))
    return scores, float(thresholds[0]), hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def full_metrics(path, original):
    scores, csv_threshold, id_digest = read_numeric_predictions(path)
    threshold = float(original.get("validation_selected_threshold", csv_threshold))
    if not np.isclose(csv_threshold, threshold, rtol=1e-8, atol=1e-8):
        raise ValueError("CSV and result thresholds differ: " + str(path))
    y = np.tile(np.array([1, 0], dtype=np.int64), len(scores))
    flat = scores.ravel()
    # Stable double-precision probability computation for calibration.
    sf = flat.astype(np.float64)
    p = np.empty_like(sf)
    pos = sf >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-sf[pos]))
    expneg = np.exp(sf[~pos])
    p[~pos] = expneg / (1.0 + expneg)
    bin_ids = np.minimum((p * 10).astype(np.int64), 9)
    ece, bins = 0.0, []
    for i in range(10):
        selected = bin_ids == i
        count = int(selected.sum())
        confidence = float(p[selected].mean()) if count else None
        positive_rate = float(y[selected].mean()) if count else None
        contribution = abs(confidence - positive_rate) * count / len(y) if count else 0.0
        ece += contribution
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": count,
                     "mean_positive_probability": confidence, "positive_frequency": positive_rate,
                     "weighted_gap": contribution})
    selected = classification_metrics(y, flat, threshold)
    fixed = classification_metrics(y, flat, 0.0)
    margin = scores[:, 0] - scores[:, 1]
    metrics = {
        "n_anchors": len(scores), "n_pairs": len(y),
        "paired_accuracy": float(np.mean(np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0)))),
        "auroc": float(roc_auc_score(y, flat)),
        "average_precision": float(average_precision_score(y, flat)),
        "mean_margin": float(margin.mean()),
        "mean_positive_probability": float(p.reshape(-1, 2)[:, 0].mean()),
        "mean_negative_probability": float(p.reshape(-1, 2)[:, 1].mean()),
        "brier_score": float(np.mean((p - y) ** 2)),
        "ece10": float(ece), "ece10_bins": bins,
        "validation_selected_threshold": threshold,
        "at_validation_threshold": selected, "at_logit_zero": fixed,
        "ordered_validation_image_id_sha256": id_digest,
    }
    aliases = {
        "balanced_accuracy_at_validation_threshold": selected["balanced_accuracy"],
        "f1_at_validation_threshold": selected["f1"],
        "positive_recall_at_validation_threshold": selected["recall"],
        "false_caption_accept_rate_at_validation_threshold": selected["fpr"],
    }
    comparisons = []
    for key, expected in original.items():
        value = aliases.get(key, metrics.get(key))
        if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
            passed = bool(np.isclose(value, expected, rtol=1e-6, atol=1e-7))
            comparisons.append({"metric": key, "original": expected, "recomputed": value,
                                "absolute_difference": abs(value - expected), "passed": passed})
    if not comparisons or not all(x["passed"] for x in comparisons):
        raise ValueError("Original metrics disagree with score-derived metrics: " + str(path)
                         + " " + json.dumps([x for x in comparisons if not x["passed"]]))
    return metrics, {"passed": True, "checks": comparisons}


def train_summary(path):
    if not path.is_file():
        return {}
    first, last, count = None, None, 0
    residuals, iterations, qmeans, qstds, entropies = [], [], [], [], []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # Writer may be appending the current final line.
        count += 1
        first = record if first is None else first
        last = record
        diag = record.get("diagnostics", record.get("ot_diagnostics", record.get("ot", {})))
        if isinstance(diag, dict):
            for target, key in ((residuals, "max_residual"), (iterations, "iterations"),
                                (qmeans, "q_mean"), (qstds, "q_std"), (entropies, "q_entropy")):
                if isinstance(diag.get(key), (int, float)):
                    target.append(diag[key])
    result = {"records": count, "last_step": (last or {}).get("step"), "last_record": last}
    if first and last and first.get("utc") and last.get("utc"):
        result["train_log_span_seconds"] = (datetime.fromisoformat(last["utc"]) - datetime.fromisoformat(first["utc"])).total_seconds()
        result["time_scope"] = "between_first_and_last_training_log_includes_any_gaps_excludes_preparation"
    if residuals:
        result["ot_summary"] = {
            "logged_diagnostic_steps": len(residuals), "max_recorded_residual": max(residuals),
            "max_recorded_iterations": max(iterations) if iterations else None,
            "mean_recorded_q_mean": float(np.mean(qmeans)) if qmeans else None,
            "mean_recorded_q_std": float(np.mean(qstds)) if qstds else None,
            "mean_recorded_q_entropy": float(np.mean(entropies)) if entropies else None,
        }
    return result


def result_row(group, budget, pipeline, manifest):
    row = {"group": group, "label": LABELS[group], "budget": budget,
           "budget_percent": int(budget), "l_images": L_COUNTS[budget],
           "l_pairs": L_COUNTS[budget] * 2, "seed": 20260825,
           "u_images": (13373 - L_COUNTS[budget]) if group in ("G3", "G4") else 0,
           "metrics": None, "metrics_verified": None, "best_step": None,
           "step": None, "target_steps": 1600, "a_source_best_step": None,
           "a_source_best_sha256": None, "stage": "original" if group == "G1" else "additional"}
    row["u_pairs"] = row["u_images"] * 2
    if group in ("G3", "G4") and budget == "100":
        row.update(state="not_applicable", reason="U=0", run_id=None)
        return row
    if group == "G1":
        root, run_id = AB_ROOT, "apple_itm_random_A_bce_l%s_s20260825" % budget
    elif group == "G2":
        root, run_id = C_ROOT, "apple_itm_random_C_from_A_best_pairusa_l%s_s20260825" % budget
    else:
        root = ROOT
        run_id = "apple_itm_%s_from_A_best_%s_l%s_s20260825" % (group, "ot" if group == "G3" else "usa_ot", budget)
    out = root / "outputs" / run_id
    result_path, status_path = out / "result.json", out / "status.json"
    result, status = read_json(result_path), read_json(status_path)
    row.update(run_id=run_id, output_directory=str(out), state=status.get("state", "pending"),
               step=status.get("step"), best_step=result.get("best_step", status.get("best_step")),
               completed_utc=result.get("completed_utc"), updated_utc=status.get("updated_utc"),
               sources={"result": str(result_path), "result_sha256": sha256(result_path),
                        "status": str(status_path), "status_sha256": sha256(status_path)})
    if result.get("state", "").startswith("completed"):
        row["state"] = result["state"]
    elif row["state"] == "pending" and pipeline.get("active_run") == run_id:
        row["state"] = "active_without_status"
    source_a = read_json(AB_ROOT / "outputs" / ("apple_itm_random_A_bce_l%s_s20260825" % budget) / "result.json")
    provenance = result.get("provenance", status.get("provenance", {}))
    run_cfg = provenance.get("base", provenance).get("run", {})
    warmstart = run_cfg.get("warmstart", run_cfg.get("model", {}).get("warmstart", {}))
    if group != "G1":
        row["a_source_best_step"] = warmstart.get("a_best_step", source_a.get("best_step"))
        row["a_source_best_sha256"] = warmstart.get("a_best_sha256")
    for key in ("elapsed_seconds", "peak_memory_allocated_mb", "peak_memory_reserved_mb", "trainable_parameters",
                "u_unique_images_seen", "u_unique_pairs_seen", "u_sample_count", "ot_nonconverged_count"):
        row[key] = result.get(key, status.get(key))
    row["active_seconds"] = result.get("active_seconds", status.get("active_seconds"))
    row["active_seconds_scope"] = "prefix_preparation_and_active_step_validation_time_excludes_model_initialization_and_checkpoint_writes"
    row["peak_gpu_bytes"] = result.get("peak_gpu_bytes", status.get("peak_gpu_bytes"))
    if row["peak_memory_allocated_mb"] is None and row["peak_gpu_bytes"] is not None:
        row["peak_memory_allocated_mb"] = row["peak_gpu_bytes"] / 1024 ** 2
    if row["u_sample_count"] is None:
        row["u_sample_count"] = result.get("u_draws", status.get("u_draws"))
    row["trainable_keys"] = result.get("initialization", {}).get("trainable_keys")
    row["last_durable_step"] = status.get("last_durable_step")
    row["physical_batch"] = status.get("physical_batch")
    row["training_log"] = train_summary(out / "train.jsonl")
    last_record = row["training_log"].get("last_record") or {}
    for key in ("u_unique_pairs_seen", "u_unique_images_seen", "physical_batch"):
        if row.get(key) is None and key in last_record:
            row[key] = last_record[key]
    if row["u_sample_count"] is None and row["training_log"].get("ot_summary"):
        row["u_sample_count"] = row["training_log"]["ot_summary"]["logged_diagnostic_steps"] * 32
        row["u_sample_count_source"] = "successful_OT_step_records_times_registered_32_queries"
    if row["step"] is None:
        row["step"] = row["training_log"].get("last_step")
    predictions = out / "best_validation_predictions.csv"
    if result.get("best_validation") and predictions.is_file():
        row["metrics"], row["metrics_verified"] = full_metrics(predictions, result["best_validation"])
        row["sources"].update(predictions=str(predictions), predictions_sha256=sha256(predictions))
    elif result.get("best_validation"):
        row["report_warning"] = "result exists but best validation predictions are missing; complete metrics not verified"
    if group in ("G3", "G4"):
        row["u_manifest_record"] = manifest.get("budgets", {}).get(budget)
    return row


def fmt(value, digits=4):
    return "—" if value is None else (str(value) if isinstance(value, (int, str)) else ("%." + str(digits) + "f") % value)


def summary_matrix(report):
    headers = ["预算", "组", "状态", "阶段步/目标", "A源best", "阶段best", "Paired acc", "AUROC", "AP", "F1(原阈值)", "Brier", "ECE10"]
    body = []
    for r in report["runs"]:
        m = r["metrics"] or {}
        body.append([str(r["budget_percent"]) + "%", r["group"], r["state"],
                     "N/A" if r["state"] == "not_applicable" else fmt(r["step"]) + "/1600",
                     fmt(r["a_source_best_step"]), fmt(r["best_step"]), fmt(m.get("paired_accuracy")),
                     fmt(m.get("auroc")), fmt(m.get("average_precision")),
                     fmt(m.get("at_validation_threshold", {}).get("f1")), fmt(m.get("brier_score")), fmt(m.get("ece10"))])
    return headers, body


def classification_matrix(report):
    keys = ["logit_threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "fpr", "fnr", "tp", "tn", "fp", "fn"]
    headers = ["预算", "组", "阈值类型", "logit阈值", "Acc", "BA", "Prec", "Rec", "F1", "Spec", "FPR", "FNR", "TP", "TN", "FP", "FN"]
    body = []
    for row in report["runs"]:
        if not row["metrics"]:
            continue
        for name, field in (("验证选择", "at_validation_threshold"), ("固定0", "at_logit_zero")):
            body.append([str(row["budget_percent"]) + "%", row["group"], name]
                        + [fmt(row["metrics"][field][key]) for key in keys])
    return headers, body


def md_table(headers, body):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |" for row in body])


def html_table(headers, body):
    return '<div class="scroll"><table><thead><tr>' + "".join("<th>" + html.escape(x) + "</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row) + "</tr>" for row in body) + "</tbody></table></div>"


def outputs(report):
    main = summary_matrix(report)
    classes = classification_matrix(report)
    extra_headers = ["预算", "组", "正负logit均差", "正对均概率", "负对均概率", "L图", "U图", "已记录训练日志跨度秒"]
    extra = []
    for r in report["runs"]:
        if not r["metrics"]:
            continue
        m = r["metrics"]
        extra.append([str(r["budget_percent"]) + "%", r["group"], fmt(m["mean_margin"]),
                      fmt(m["mean_positive_probability"]), fmt(m["mean_negative_probability"]),
                      r["l_images"], r["u_images"], fmt(r["training_log"].get("train_log_span_seconds"), 1)])
    ot_headers = ["预算", "组", "L图/U图", "已访问U图/对", "U抽样次数", "活动耗时秒", "训练峰值MiB", "OT诊断步数", "最大残差", "最大迭代", "平均q", "平均q标准差", "平均q熵"]
    ot_rows = []
    for r in report["runs"]:
        if r["group"] not in ("G3", "G4") or r["state"] == "not_applicable":
            continue
        diag = r.get("training_log", {}).get("ot_summary", {})
        ot_rows.append([str(r["budget_percent"]) + "%", r["group"], str(r["l_images"]) + "/" + str(r["u_images"]),
                        fmt(r.get("u_unique_images_seen")) + "/" + fmt(r.get("u_unique_pairs_seen")),
                        fmt(r.get("u_sample_count")), fmt(r.get("active_seconds"), 1),
                        fmt(r.get("peak_memory_allocated_mb"), 1), fmt(diag.get("logged_diagnostic_steps")),
                        "—" if diag.get("max_recorded_residual") is None else "%.3g" % diag["max_recorded_residual"],
                        fmt(diag.get("max_recorded_iterations")), fmt(diag.get("mean_recorded_q_mean")),
                        fmt(diag.get("mean_recorded_q_std")), fmt(diag.get("mean_recorded_q_entropy"))])
    status = "新增 OT 已完成 %d/10；已有 G1/G2 完成 %d/12；本表有效结果单元 %d/22。" % (report["new_completed"], report["existing_completed"], report["completed_total"])
    md = ["# 最新四组消融结果", "", "生成时间：" + report["generated_at"], "", status,
          "", "新队列状态：`" + str(report["pipeline_state"]) + "`。", "", "## 结果与进度", "",
          md_table(*main), "", "## 完整二分类指标", "", md_table(*classes),
          "", "## 概率、数据量与日志时间", "", md_table(extra_headers, extra), "",
          "时间列仅为首末训练日志时间差，包含间隔且不包含准备阶段，不冒称端到端训练耗时。", "",
          "## OT 利用与运行诊断", "", md_table(ot_headers, ot_rows), "",
          "活动耗时累计前缀预热与训练/验证，未包含模型初始化和检查点写盘；峰值显存为前缀准备结束后训练阶段的最大 allocated bytes。缺测项为 —。", "",
          "## 解释范围", ""] + ["- " + x for x in NOTES]
    md += ["", "## 数值来源核验", "", "已重算并通过原 result 数值核验：%d 组。" % report["metrics_verified_count"],
           "源文件路径、SHA256、全部精度指标、混淆矩阵、ECE 分桶及逐指标核验见 `latest_progress.json`。", ""]
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>苹果 ITM 四组消融结果</title><style>
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#1d2939;background:#f4f6f9}body{margin:0;padding:30px}main{max-width:1480px;margin:auto}h1{font-size:28px;margin-bottom:8px}h2{font-size:20px;margin-top:30px}p,li{line-height:1.7}.muted{color:#667085}.card{background:#fff;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:18px 0}.scroll{overflow:auto;border:1px solid #e2e8f0;border-radius:8px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:13px}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #edf0f4}th{background:#eaf0f8;color:#24416b}th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}tr:nth-child(even){background:#f8fafc}.badge{font-size:20px;font-weight:600;color:#224e79}details{margin:12px 0}code{word-break:break-all} @media print{body{padding:0;background:white}.scroll{overflow:visible}table{font-size:8px}th,td{padding:4px}}
</style></head><body><main>"""
    document += "<h1>苹果图文匹配 · 四组消融</h1><p class='muted'>" + html.escape(report["generated_at"]) + " · 单种子 · 固定验证集</p>"
    document += "<div class='card'><p class='badge'>" + html.escape(status) + "</p><p>新队列状态：" + html.escape(str(report["pipeline_state"])) + "；数值来源核验通过 " + str(report["metrics_verified_count"]) + " 组。</p></div>"
    document += "<h2>结果与进度</h2>" + html_table(*main)
    document += "<h2>完整二分类指标</h2><p class='muted'>原验证阈值与固定 logit=0 分开列出；正例为图文匹配。</p>" + html_table(*classes)
    document += "<h2>概率、数据量与日志时间</h2>" + html_table(extra_headers, extra)
    document += "<p class='muted'>日志跨度包含间隔、不包含准备阶段；详细记录和完整精度指标保存在同目录 latest_progress.json。</p>"
    document += "<h2>OT 利用与运行诊断</h2>" + html_table(ot_headers, ot_rows)
    document += "<p class='muted'>活动耗时累计前缀预热与训练/验证，未包含模型初始化和检查点写盘；显存统计为预热后训练阶段峰值。缺测项为 —。</p>"
    document += "<h2>解释范围</h2><div class='card'><ul>" + "".join("<li>" + html.escape(x) + "</li>" for x in NOTES) + "</ul></div>"
    document += "<details><summary>逐组来源</summary><ul>" + "".join("<li>" + html.escape(r["group"] + " " + str(r["budget_percent"]) + "%: " + str(r.get("run_id"))) + "<br><code>" + html.escape(str(r.get("output_directory", "N/A"))) + "</code></li>" for r in report["runs"]) + "</ul></details></main></body></html>"
    return "\n".join(md), document


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    pipeline = read_json(ROOT / "outputs" / "pipeline_state.json")
    manifest = read_json(ROOT / "data" / "manifest.json")
    runs = [result_row(group, budget, pipeline, manifest) for budget in BUDGETS for group in LABELS]
    complete = [x for x in runs if x["state"].startswith("completed")]
    verified = [x for x in runs if x["metrics_verified"] and x["metrics_verified"]["passed"]]
    id_hashes = {x["metrics"]["ordered_validation_image_id_sha256"] for x in verified}
    if len(id_hashes) > 1:
        raise ValueError("Runs do not share the same ordered validation image IDs")
    report = {
        "schema_version": 1, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_root": str(ROOT), "pipeline_state": pipeline.get("state", "not_started"),
        "pipeline": pipeline, "new_total": 10, "existing_total": 12, "effective_total": 22,
        "new_completed": sum(x["group"] in ("G3", "G4") for x in complete),
        "existing_completed": sum(x["group"] in ("G1", "G2") for x in complete),
        "completed_total": len(complete), "metrics_verified_count": len(verified),
        "not_applicable_cells": 2, "notes": NOTES,
        "source_manifest_sha256": sha256(ROOT / "data" / "manifest.json"),
        "report_code_sha256": sha256(Path(__file__)), "runs": runs,
    }
    md, document = outputs(report)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(args.reports_dir / "latest_progress.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(args.reports_dir / "最新四组消融结果.md", md)
    atomic_write(args.reports_dir / "最新四组消融结果.html", document)
    print(json.dumps({"new_completed": report["new_completed"], "new_total": 10,
                      "existing_completed": report["existing_completed"], "metrics_verified_count": len(verified),
                      "reports": str(args.reports_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
