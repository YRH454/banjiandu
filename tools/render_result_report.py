"""Render a private, self-contained HTML report from a completed 40-run summary.

Only aggregate metrics are embedded. No provenance, input rows, paths to model
assets, predictions, or server connection details are copied into the HTML.
The report is printed to stdout so the caller controls where it is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATASETS = {"apple": "苹果", "cassava": "木薯", "rice": "水稻", "banana": "香蕉"}
METHODS = {"softmatch": "SoftMatch", "simmatch": "SimMatch"}
BUDGETS = (1, 5, 10, 20, 30)
RUN_PATTERN = re.compile(r"^(apple|cassava|rice|banana)_(001|005|010|020|030)_(softmatch|simmatch)_s20260825$")


def validate_metrics(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (cm[k] for k in ("tn", "fp", "fn", "tp"))
    if any(not isinstance(x, int) or x < 0 for x in (tn, fp, fn, tp)):
        raise ValueError("Invalid confusion-matrix counts")
    n = tn + fp + fn + tp
    div = lambda a, b: a / b if b else 0.0
    calculated = {
        "accuracy": div(tp + tn, n), "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn), "negative_recall": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "macro_f1": (div(2 * tp, 2 * tp + fp + fn) + div(2 * tn, 2 * tn + fp + fn)) / 2,
        "balanced_accuracy": (div(tp, tp + fn) + div(tn, tn + fp)) / 2,
    }
    if metrics["n"] != n:
        raise ValueError("Confusion matrix does not match validation count")
    for key, expected in calculated.items():
        if not math.isclose(metrics[key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Confusion matrix does not match {key}")
    for key in ("accuracy", "precision", "recall", "negative_recall", "f1", "macro_f1", "balanced_accuracy", "auroc", "average_precision", "brier", "ece_10"):
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key}")


def load_rows(summary: dict) -> list[dict]:
    results = summary["results"]
    if summary["configurations"] != 40 or len(results) != 40:
        raise ValueError("The report requires all 40 completed configurations")
    if summary["single_seed"] != 20260825 or not summary["no_Test"] or summary["equal_compute_claim"]:
        raise ValueError("Unexpected scientific reporting scope")
    rows, seen = [], set()
    for run_id, result in results.items():
        match = RUN_PATTERN.fullmatch(run_id)
        if match is None:
            raise ValueError(f"Unexpected run ID: {run_id}")
        dataset, budget_code, method = match.groups()
        budget = int(budget_code)
        key = dataset, budget, method
        if key in seen:
            raise ValueError("Duplicate configuration")
        seen.add(key)
        target = 2200 if method == "softmatch" else 2400
        if result["run_id"] != run_id or result["state"] != "completed" or result["successful_steps"] != target or result["test_evaluated"]:
            raise ValueError(f"Incomplete or inconsistent run: {run_id}")
        metrics = result["best_metrics"]
        validate_metrics(metrics)
        validate_metrics(metrics["threshold_0_5"])
        if metrics["step"] != result["best_step"] or metrics["evaluation_model"] != "ema":
            raise ValueError("Unexpected checkpoint or evaluation model")
        if metrics["n"] != 800 or metrics["validation_anchors"] != 400 or metrics["threshold_0_5"]["threshold"] != 0.5:
            raise ValueError("Unexpected validation protocol")
        rows.append({
            "run_id": run_id, "dataset": dataset, "budget": budget, "method": method,
            "steps": target, "best_step": result["best_step"], "metrics": metrics,
            "wall_seconds": result["wall_seconds_including_validation_and_checkpointing"],
            "training_seconds": result["training_step_seconds"],
        })
    expected = {(d, b, m) for d in DATASETS for b in BUDGETS for m in METHODS}
    if seen != expected:
        raise ValueError("Configuration grid is incomplete")
    return sorted(rows, key=lambda r: (list(DATASETS).index(r["dataset"]), r["budget"], list(METHODS).index(r["method"])))


def render(summary_path: Path) -> str:
    raw = summary_path.read_bytes()
    summary = json.loads(raw)
    rows = load_rows(summary)
    means = {m: statistics.fmean(r["metrics"]["auroc"] for r in rows if r["method"] == m) for m in METHODS}
    pairs = {(r["dataset"], r["budget"], r["method"]): r for r in rows}
    wins = sum(pairs[d, b, "simmatch"]["metrics"]["auroc"] > pairs[d, b, "softmatch"]["metrics"]["auroc"] for d in DATASETS for b in BUDGETS)
    exceptions = [f"{DATASETS[d]} {b}%" for d in DATASETS for b in BUDGETS if pairs[d, b, "simmatch"]["metrics"]["auroc"] < pairs[d, b, "softmatch"]["metrics"]["auroc"]]
    completed = datetime.fromisoformat(summary["completed_utc"]).astimezone(timezone(timedelta(hours=8)))
    initial_rows = []
    for r in rows:
        m = r["metrics"]
        initial_rows.append(
            '<tr><td>' + DATASETS[r["dataset"]] + '</td><td>' + str(r["budget"]) + '%</td><td><span class="tag ' + r["method"] + '">' + METHODS[r["method"]] + '</span></td>'
            + ''.join(f'<td class="num">{m[k]:.4f}</td>' if k == "auroc" else f'<td class="num">{m[k] * 100:.2f}%</td>' for k in ("auroc", "paired_accuracy", "accuracy", "macro_f1", "f1", "precision", "recall"))
            + f'<td class="num">{m["threshold"]:.4f}</td><td class="num">{r["best_step"]}/{r["steps"]}</td><td><button class="detail" data-run="{r["run_id"]}">详情</button></td></tr>'
        )
    embedded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    values = {
        "___DATA___": embedded, "___ROWS___": "\n".join(initial_rows),
        "___COMPLETED___": completed.strftime("%Y-%m-%d %H:%M:%S 北京时间"),
        "___SOFT_MEAN___": f'{means["softmatch"]:.4f}', "___SIM_MEAN___": f'{means["simmatch"]:.4f}',
        "___DELTA___": f'{means["simmatch"] - means["softmatch"]:+.4f}',
        "___WINS___": str(wins), "___EXCEPTIONS___": html.escape("、".join(exceptions)),
        "___SOURCE___": html.escape(summary_path.name), "___HASH___": hashlib.sha256(raw).hexdigest(),
    }
    output = TEMPLATE
    for key, value in values.items():
        output = output.replace(key, value)
    return output


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="description" content="40组SoftMatch与SimMatch图文匹配实验的本地验证集报告。单种子，未评估Test，非等算力。">
<title>40组实验结果 · SoftMatch / SimMatch</title>
<style>
:root{--bg:#f3f5f7;--paper:#fff;--ink:#182537;--muted:#596779;--line:#dce2e8;--soft:#3e66a6;--sim:#007a68;--warning:#775210}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}button,select{font:inherit}a{color:#365e97}button{cursor:pointer}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #e6ae4d;outline-offset:3px}header,main,footer{max-width:1320px;margin:auto;padding:0 28px}header{padding-top:42px;padding-bottom:24px}.eyebrow{font-size:12px;letter-spacing:.08em;color:var(--muted)}h1{font-size:clamp(27px,4vw,39px);line-height:1.3;margin:9px 0 13px}h2{font-size:22px;margin:0 0 8px}h3{font-size:17px;margin:0}.lead{margin:0;color:var(--muted);max-width:930px}.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:17px}.chip{border:1px solid var(--line);border-radius:5px;padding:3px 9px;background:var(--paper);font-size:12px}nav{display:flex;gap:22px;margin-top:18px;flex-wrap:wrap}nav a{text-decoration:none;font-size:14px}section{margin-bottom:30px;scroll-margin-top:18px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:22px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:20px}.card-label{font-size:13px;color:var(--muted)}.big{font-size:30px;font-weight:650;line-height:1.4;font-variant-numeric:tabular-nums;margin:6px 0}.sub{color:var(--muted);font-size:12px}.warning{border-left:4px solid #c18b2b;background:#fff9ef;padding:14px 18px;color:var(--warning);border-radius:4px;margin:18px 0 22px}.warning p{margin:4px 0}.note{font-size:13px;color:var(--muted);margin:7px 0}.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin:17px 0}.controls label{display:grid;gap:4px;font-size:12px;color:var(--muted)}select{padding:7px 30px 7px 9px;border:1px solid #bdc8d3;border-radius:5px;background:white;color:var(--ink)}.btn{background:white;border:1px solid #bdc8d3;border-radius:5px;padding:7px 13px;color:var(--ink)}.btn.primary{background:var(--ink);color:#fff;border-color:var(--ink)}.legend{display:flex;gap:18px;font-size:13px;margin:6px 0 12px;flex-wrap:wrap}.soft-text{color:var(--soft)}.sim-text{color:var(--sim)}.charts{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.chart{border:1px solid var(--line);padding:16px;border-radius:6px}.chart svg{display:block;width:100%;height:auto}.chart small{color:var(--muted)}.scroll{overflow:auto;border:1px solid var(--line);border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 11px;text-align:left;border-bottom:1px solid #e5e9ee;white-space:nowrap}thead{background:#f0f3f6}th{font-weight:600}tbody tr:hover{background:#f5f8fa}tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums}.group-start td{border-top:2px solid #c5cfd8}.tag{font-size:12px;font-weight:600}.tag.softmatch{color:var(--soft)}.tag.simmatch{color:var(--sim)}.positive{color:var(--sim);font-weight:600}.negative{color:var(--soft);font-weight:600}.detail{border:0;background:transparent;color:#365e97;padding:2px 5px;text-decoration:underline}.table-caption{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:10px}.interpretation{display:grid;grid-template-columns:1fr 1fr;gap:18px}.interpretation p{margin:8px 0 0}.definitions{display:grid;grid-template-columns:1fr 1fr;gap:14px 28px}.definitions dt{font-weight:600}.definitions dd{margin:3px 0 0;color:var(--muted);font-size:13px}footer{font-size:12px;color:var(--muted);padding-top:5px;padding-bottom:35px}.hash{overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace;font-size:11px}dialog{border:1px solid var(--line);border-radius:9px;padding:25px;width:min(780px,calc(100vw - 28px));color:var(--ink);max-height:90vh;overflow:auto}dialog::backdrop{background:#18253780}.dialog-head{display:flex;align-items:start;justify-content:space-between;gap:15px}.dialog-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:18px}.cm th,.cm td{white-space:normal;text-align:center}.cm td{font-size:20px;font-variant-numeric:tabular-nums}.cm .diag{background:#edf7f2}.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px 14px;font-size:13px}.stats dt{color:var(--muted)}.stats dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.empty{padding:24px;text-align:center;color:var(--muted)}.status{font-size:12px;color:var(--muted)}noscript{display:block;background:#fff9ef;padding:12px;margin:12px 0}
@media(max-width:850px){.cards{grid-template-columns:repeat(2,1fr)}header,main,footer{padding-left:16px;padding-right:16px}.panel{padding:17px}.interpretation,.definitions{grid-template-columns:1fr}}
@media(max-width:560px){.charts,.dialog-grid{grid-template-columns:1fr}.big{font-size:25px}.card{padding:14px}header{padding-top:25px}.controls{gap:9px}.controls label{flex:1;min-width:130px}select{width:100%}}
@media print{@page{size:A4 landscape;margin:12mm}body{background:white;font-size:11px}header,main,footer{max-width:none;padding-left:0;padding-right:0}header{padding-top:0}h1{font-size:25px}nav,.controls,.detail,.print-btn,dialog{display:none!important}.panel,.card,.chart{border-radius:0}.panel{padding:13px}.cards{gap:9px}.big{font-size:25px}.scroll{overflow:visible}th,td{font-size:8px;padding:5px 4px}.chart,.card,.warning,.interpretation{break-inside:avoid}.hash{font-size:9px}.charts{grid-template-columns:1fr 1fr}.warning{background:white}section{margin-bottom:18px}thead{display:table-header-group}tr{break-inside:avoid}}
</style>
</head>
<body>
<header>
<div class="eyebrow">本地实验报告 · 图文匹配二分类（ITM适配）</div>
<h1>SoftMatch 与 SimMatch：40组实验结果</h1>
<p class="lead">苹果、木薯、水稻、香蕉 × 1%、5%、10%、20%、30% 标注预算 × 两种算法。展示每组 <strong>best checkpoint 的 EMA 验证结果</strong>，不是病害分类成绩，也不是独立测试集成绩。</p>
<div class="chips"><span class="chip">40 / 40 完成</span><span class="chip">种子 20260825 · 单种子</span><span class="chip">400验证锚图 / 800图文对</span><span class="chip">未评估 Test</span><span class="chip">非等算力</span><span class="chip">___COMPLETED___</span></div>
<nav aria-label="页面章节"><a href="#overview">概览</a><a href="#curves">趋势图</a><a href="#comparison">同预算对比</a><a href="#all-runs">40组明细</a><a href="#reading">阅读口径</a><button class="btn print-btn" id="print">打印 / 保存PDF</button></nav>
</header>
<main>
<noscript>当前未启用JavaScript。下方40组完整默认明细仍可阅读；图表、筛选、详情和CSV导出需要启用JavaScript。</noscript>
<section id="overview">
<div class="cards">
<div class="card"><div class="card-label">配置完成情况</div><div class="big">40 / 40</div><div class="sub">SoftMatch 20组 · SimMatch 20组<br>4个数据集，每算法5个预算</div></div>
<div class="card"><div class="card-label">SoftMatch · 20配置AUROC简单均值</div><div class="big soft-text">___SOFT_MEAN___</div><div class="sub">每组2200次成功更新<br>不是合并样本的全局AUROC</div></div>
<div class="card"><div class="card-label">SimMatch · 20配置AUROC简单均值</div><div class="big sim-text">___SIM_MEAN___</div><div class="sub">每组2400次成功更新<br>相对SoftMatch均值差 ___DELTA___</div></div>
<div class="card"><div class="card-label">同数据集、同预算的AUROC观测比较</div><div class="big">___WINS___ / 20</div><div class="sub">SimMatch数值更高的配对数<br>不代表显著性检验或普遍优越性</div></div>
</div>
<div class="warning"><p><strong>先看口径，再看数值。</strong> 只有一个训练种子，没有均值±标准差或显著性检验。两算法更新预算不同，不声称等算力。阈值在验证集上按 macro-F1 选择，best 权重也由验证集选择，因此这些分数不能代替独立测试表现。</p></div>
<div class="interpretation">
<div class="panel"><h3>这轮数据的直接观察</h3><p>SimMatch在 ___WINS___ / 20 个同数据集、同预算配置中具有更高AUROC。SoftMatch数值更高的例外是 <strong>___EXCEPTIONS___</strong>；对比表保留这些例外。</p><p class="note">这是本轮观测，不是“SimMatch在所有情况下都更好”的结论。</p></div>
<div class="panel"><h3>best step 不等于最终进度</h3><p>40组均完成目标更新。表中的 best step 是被选中权重所在的历史步骤；即使best step很早，也不表示训练提前停止。配对准确率（Paired Acc）优先、AUROC次优用于选择best权重。</p><p class="note">SoftMatch总计44,000次更新；SimMatch总计48,000次更新。</p></div>
</div>
</section>
<section id="curves" class="panel">
<h2>不同标注预算的结果趋势</h2><p class="note">每张图一个数据集；横轴是五个预算类别，不是连续比例坐标。所有图使用相同纵轴范围，分数越高越好。</p>
<div class="controls"><label>图表 / 对比指标<select id="chart-metric"><option value="auroc">AUROC</option><option value="paired_accuracy">配对准确率 Paired Acc</option><option value="macro_f1">Macro-F1</option><option value="accuracy">图文对准确率 Accuracy</option></select></label><label>阈值口径（同时影响明细）<select id="threshold"><option value="selected">验证集选择的阈值</option><option value="fixed">固定阈值0.5</option></select></label></div>
<div class="legend"><span class="soft-text">● ━ SoftMatch</span><span class="sim-text">◆ ━ SimMatch</span><span>灰色虚线：0.5参考线（仅AUROC / 准确率）</span></div>
<div id="charts" class="charts"></div>
</section>
<section id="comparison" class="panel"><h2>同数据集、同预算：20组配对比较</h2><p class="note">Δ = SimMatch − SoftMatch。表中指标跟随上面的指标和阈值选择；训练更新预算仍不同。</p><div class="scroll"><table><thead><tr><th>数据集</th><th>标注预算</th><th class="num">SoftMatch</th><th class="num">SimMatch</th><th class="num">Δ（0–1单位）</th><th>本轮数值较高</th></tr></thead><tbody id="pair-body"></tbody></table></div></section>
<section id="all-runs" class="panel"><h2>全部40组：筛选、排序与详情</h2><p class="note">F1为正类F1；Macro-F1同时平均正、负两类。点击“详情”查看AP、校准误差、混淆矩阵和耗时。</p>
<div class="controls"><label>数据集<select id="dataset"><option value="all">全部数据集</option><option value="apple">苹果</option><option value="cassava">木薯</option><option value="rice">水稻</option><option value="banana">香蕉</option></select></label><label>算法<select id="method"><option value="all">全部算法</option><option value="softmatch">SoftMatch</option><option value="simmatch">SimMatch</option></select></label><label>标注预算<select id="budget"><option value="all">全部预算</option><option value="1">1%</option><option value="5">5%</option><option value="10">10%</option><option value="20">20%</option><option value="30">30%</option></select></label><label>排序<select id="sort"><option value="original">实验定义顺序</option><option value="auroc">AUROC降序</option><option value="paired_accuracy">Paired Acc降序</option><option value="macro_f1">Macro-F1降序</option></select></label><button class="btn" id="reset">重置筛选</button><button class="btn primary" id="export">导出当前筛选CSV</button></div>
<div class="table-caption"><span id="count" class="status" aria-live="polite">显示40 / 40组</span><span class="status">均为EMA best checkpoint；数值由原始汇总读取</span></div>
<div class="scroll"><table id="runs-table"><thead><tr><th>数据集</th><th>预算</th><th>算法</th><th class="num">AUROC</th><th class="num">Paired Acc</th><th class="num">Accuracy</th><th class="num">Macro-F1</th><th class="num">F1（正）</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">阈值</th><th class="num">best / 最终步数</th><th>详情</th></tr></thead><tbody id="run-body">___ROWS___</tbody></table></div>
</section>
<section id="reading" class="panel"><h2>指标与阅读口径</h2>
<dl class="definitions">
<div><dt>AUROC（越高越好）</dt><dd>匹配正、负图文对的排序区分能力，取值0–1；与当前分类阈值无关。0.5为随机排序参考，不是模型质量达标线。</dd></div>
<div><dt>Paired Acc（越高越好）</dt><dd>同一验证锚图的正caption分数高于其负caption分数的比例，取值0–100%。它不是800个图文对的分类准确率。</dd></div>
<div><dt>Accuracy / F1 / Macro-F1</dt><dd>基于当前阈值的二分类指标。F1仅表示“匹配”正类；Macro-F1平均正、负两类F1。不能把它们解释成四种作物或病害的分类准确率。</dd></div>
<div><dt>两个阈值口径</dt><dd>默认阈值在同一验证集上最大化macro-F1，平局偏向0.5；另提供固定0.5结果。切换阈值不会改变AUROC、AP和Paired Acc，也不会重新选择checkpoint。</dd></div>
<div><dt>AP / Brier / ECE-10</dt><dd>AP为平均精度，越高越好；Brier为预测概率的均方误差，ECE-10为10个概率分箱的校准误差，二者越低越好，均与分类阈值无关。</dd></div>
<div><dt>时间与复现边界</dt><dd>每组耗时含验证与checkpoint保存，多组可能同时运行，不能相加当作项目壁钟耗时。不同运行阶段有并行和缓存差异，不能直接用这些耗时声称算法速度优势。</dd></div>
</dl><p class="note">本报告只嵌入汇总指标，不含数据行、样本ID、caption、预测分数、权重、服务器连接信息或完整provenance。HTML与原始实验结果留在本地，不上传公开代码仓库。</p>
</section>
</main>
<dialog id="detail-dialog" aria-labelledby="detail-title"><div class="dialog-head"><div><h2 id="detail-title"></h2><p id="detail-run" class="note"></p></div><button class="btn" id="close-dialog" aria-label="关闭详情">关闭</button></div><div id="detail-content"></div></dialog>
<footer><div>来源：___SOURCE___ · 实验完成：___COMPLETED___ · 已核对40个唯一配置、目标更新次数及各混淆矩阵与指标的一致性。</div><div class="hash">原始汇总 SHA256：___HASH___</div><div>这是一个无外部依赖的本地HTML；可离线打开、筛选、查看详情和导出当前指标。</div></footer>
<script type="application/json" id="report-data">___DATA___</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const DS = {apple:'苹果',cassava:'木薯',rice:'水稻',banana:'香蕉'};
const METHODS = {softmatch:'SoftMatch',simmatch:'SimMatch'};
const LABEL = {auroc:'AUROC',paired_accuracy:'Paired Acc',macro_f1:'Macro-F1',accuracy:'Accuracy'};
const BUDGETS = [1,5,10,20,30];
const $ = id => document.getElementById(id);
const esc = text => String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = x => Number.isFinite(x) ? x.toFixed(4) : '—';
const pct = x => Number.isFinite(x) ? (100*x).toFixed(2)+'%' : '—';
function metrics(r){if($('threshold').value === 'fixed')return {...r.metrics.threshold_0_5,paired_accuracy:r.metrics.paired_accuracy};return r.metrics;}
function visibleRows(){const rows=DATA.filter(r=>($('dataset').value==='all'||r.dataset===$('dataset').value)&&($('method').value==='all'||r.method===$('method').value)&&($('budget').value==='all'||r.budget===Number($('budget').value)));const order=$('sort').value;return order==='original'?rows:rows.sort((a,b)=>metrics(b)[order]-metrics(a)[order]);}
function renderRows(){const rows=visibleRows();$('run-body').innerHTML=rows.length?rows.map(r=>{const m=metrics(r);return `<tr><td>${DS[r.dataset]}</td><td>${r.budget}%</td><td><span class="tag ${r.method}">${METHODS[r.method]}</span></td>${['auroc','paired_accuracy','accuracy','macro_f1','f1','precision','recall'].map(k=>`<td class="num">${k==='auroc'?num(m[k]):pct(m[k])}</td>`).join('')}<td class="num">${num(m.threshold)}</td><td class="num">${r.best_step}/${r.steps}</td><td><button class="detail" data-run="${esc(r.run_id)}" aria-label="查看${DS[r.dataset]}${r.budget}%${METHODS[r.method]}详情">详情</button></td></tr>`;}).join(''):'<tr><td colspan="13" class="empty">当前条件没有记录。</td></tr>';$('count').textContent=`显示${rows.length} / 40组 · ${$('threshold').value==='fixed'?'固定阈值0.5':'验证集选择的阈值'}`;}
function score(r,key){return metrics(r)[key];}
function renderCharts(){const key=$('chart-metric').value;const low=Math.min(...DATA.map(r=>score(r,key)))<0.4?0:0.4;const top=1;const x=i=>52+i*94;const y=v=>178-(v-low)/(top-low)*151;const ticks=[0,1,2,3,4].map(i=>low+(top-low)*i/4);$('charts').innerHTML=Object.entries(DS).map(([ds,title])=>{let svg=ticks.map(v=>`<line x1="52" y1="${y(v)}" x2="428" y2="${y(v)}" stroke="#dfe5ea"/><text x="44" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#596779">${v.toFixed(2)}</text>`).join('');if(['auroc','accuracy','paired_accuracy'].includes(key))svg+=`<line x1="52" y1="${y(0.5)}" x2="428" y2="${y(0.5)}" stroke="#929ca8" stroke-dasharray="4 4"/>`;for(const method of Object.keys(METHODS)){const rows=BUDGETS.map(b=>DATA.find(r=>r.dataset===ds&&r.budget===b&&r.method===method));const color=method==='softmatch'?'#3e66a6':'#007a68';svg+=`<polyline points="${rows.map((r,i)=>`${x(i)},${y(score(r,key))}`).join(' ')}" stroke="${color}" stroke-width="2.3" fill="none"/>`;svg+=rows.map((r,i)=>{const title=`${METHODS[method]} ${r.budget}%：${num(score(r,key))}`;return method==='softmatch'?`<circle cx="${x(i)}" cy="${y(score(r,key))}" r="4" fill="${color}"><title>${title}</title></circle>`:`<path d="M ${x(i)} ${y(score(r,key))-5} l 5 5 -5 5 -5 -5 Z" fill="${color}"><title>${title}</title></path>`;}).join('');}svg+=BUDGETS.map((b,i)=>`<text x="${x(i)}" y="199" text-anchor="middle" font-size="11" fill="#596779">${b}%</text>`).join('');return `<div class="chart"><h3>${title} <span class="note">/ ${LABEL[key]}</span></h3><svg viewBox="0 0 480 216" role="img" aria-label="${title}两算法随标注预算变化的${LABEL[key]}比较">${svg}</svg><small>纵轴 ${low.toFixed(2)}–1.00 · 预算为分类轴 · hover点位可读原值</small></div>`;}).join('');renderPairs();}
function renderPairs(){const key=$('chart-metric').value;let out='';for(const ds of Object.keys(DS)){for(const b of BUDGETS){const soft=DATA.find(r=>r.dataset===ds&&r.budget===b&&r.method==='softmatch');const sim=DATA.find(r=>r.dataset===ds&&r.budget===b&&r.method==='simmatch');const a=score(soft,key),z=score(sim,key),delta=z-a;out+=`<tr${b===1?' class="group-start"':''}><td>${DS[ds]}</td><td>${b}%</td><td class="num soft-text">${num(a)}</td><td class="num sim-text">${num(z)}</td><td class="num ${delta>0?'positive':delta<0?'negative':''}">${delta>0?'+':''}${num(delta)}</td><td>${Math.abs(delta)<1e-12?'相同':delta>0?'SimMatch':'SoftMatch'}</td></tr>`;}}$('pair-body').innerHTML=out;}
function showDetail(run){const r=DATA.find(r=>r.run_id===run);if(!r)return;const m=metrics(r),cm=m.confusion_matrix;$('detail-title').textContent=`${DS[r.dataset]} ${r.budget}% · ${METHODS[r.method]}`;$('detail-run').textContent=r.run_id;const extra=[['阈值',num(m.threshold)],['AUROC',num(m.auroc)],['Paired Acc',pct(m.paired_accuracy)],['Accuracy',pct(m.accuracy)],['Macro-F1',pct(m.macro_f1)],['F1（正类）',pct(m.f1)],['Precision',pct(m.precision)],['Recall（正类）',pct(m.recall)],['Recall（负类）',pct(m.negative_recall)],['Balanced Accuracy',pct(m.balanced_accuracy)],['AP',num(m.average_precision)],['Brier ↓',num(m.brier)],['ECE-10 ↓',num(m.ece_10)]];$('detail-content').innerHTML=`<div class="dialog-grid"><div><h3>当前阈值的混淆矩阵</h3><p class="note">行：真实类别；列：预测类别。正类=图文匹配，负类=错配。</p><table class="cm"><thead><tr><th></th><th>预测负</th><th>预测正</th></tr></thead><tbody><tr><th>真实负</th><td class="diag">TN ${cm.tn}</td><td>FP ${cm.fp}</td></tr><tr><th>真实正</th><td>FN ${cm.fn}</td><td class="diag">TP ${cm.tp}</td></tr></tbody></table><p class="note">400验证锚图，共${m.n}对。当前口径：${$('threshold').value==='fixed'?'固定0.5':'验证集选择阈值'}。</p></div><div><h3>完整指标</h3><dl class="stats">${extra.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></div></div><hr style="border:0;border-top:1px solid #dce2e8"><p class="note">已完成 ${r.steps} 次更新；best checkpoint 在第 ${r.best_step} 步。EMA验证，无Test推理。</p><p class="note">本组壁钟耗时 ${(r.wall_seconds/60).toFixed(1)} 分钟（含验证与保存）；累计训练步用时 ${(r.training_seconds/60).toFixed(1)} 分钟。该耗时不是公平速度对照，也不能与并行组相加作为项目总时长。</p>`;$('detail-dialog').showModal();}
function exportCSV(){const headers=['数据集','标注预算(%)','算法','AUROC','PairedAccuracy','Accuracy','MacroF1','PositiveF1','Precision','Recall','NegativeRecall','AP','Brier','ECE10','Threshold','BestStep','SuccessfulSteps','WallSeconds','RunID'];const quote=v=>'"'+String(v).replace(/"/g,'""')+'"';const lines=[headers.map(quote).join(',')];for(const r of visibleRows()){const m=metrics(r);const vals=[DS[r.dataset],r.budget,METHODS[r.method],m.auroc,m.paired_accuracy,m.accuracy,m.macro_f1,m.f1,m.precision,m.recall,m.negative_recall,m.average_precision,m.brier,m.ece_10,m.threshold,r.best_step,r.steps,r.wall_seconds,r.run_id];lines.push(vals.map(quote).join(','));}const url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`实验结果_${$('threshold').value}_${visibleRows().length}组.csv`;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
for(const id of ['dataset','method','budget','sort'])$(id).addEventListener('change',renderRows);
$('chart-metric').addEventListener('change',renderCharts);
$('threshold').addEventListener('change',()=>{renderRows();renderCharts();});
$('reset').addEventListener('click',()=>{for(const id of ['dataset','method','budget'])$(id).value='all';$('sort').value='original';renderRows();});
$('run-body').addEventListener('click',e=>{const button=e.target.closest('.detail');if(button)showDetail(button.dataset.run);});
$('close-dialog').addEventListener('click',()=>$('detail-dialog').close());
$('detail-dialog').addEventListener('click',e=>{if(e.target===$('detail-dialog')){const r=$('detail-dialog').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('detail-dialog').close();}});
$('export').addEventListener('click',exportCSV);
$('print').addEventListener('click',()=>window.print());
renderRows();renderCharts();
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    sys.stdout.buffer.write(render(args.summary).encode("utf-8"))


if __name__ == "__main__":
    main()
