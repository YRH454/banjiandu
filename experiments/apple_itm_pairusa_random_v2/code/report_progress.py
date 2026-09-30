"""Summarise completed validation runs without loading models or the test set."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None

def main():
    state = read(ROOT / 'outputs/pipeline_state.json') or {'state':'preparing_not_started'}
    gate = read(ROOT / 'audit/negative_quality_gate.json')
    if gate and gate.get('scientific_stop_requires_user_decision') and state['state'] == 'preparing_not_started':
        state['state'] = 'scientific_quality_stop_before_training'
    active_status = None
    latest_train = None
    if state.get('active_run'):
        active_dir = ROOT / 'outputs' / state['active_run']
        active_status = read(active_dir / 'status.json')
        if (active_dir / 'train.jsonl').is_file():
            lines = (active_dir / 'train.jsonl').read_text(encoding='utf-8').splitlines()
            for line in reversed(lines):
                try:
                    latest_train = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    results = []
    for path in sorted((ROOT / 'outputs').glob('apple_itm_*/result.json')):
        if '_smoke' in path.parent.name:
            continue
        item = read(path)
        provenance = item.get('provenance',{})
        cfg = provenance.get('base',provenance).get('run',{})
        results.append({'run_id':item['run_id'],'budget':cfg.get('budget'),
                        'method':cfg.get('method'),'best_step':item.get('best_step'),
                        'validation':item.get('best_validation',{}),'result_path':str(path)})
    report = {'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'pipeline_state':state.get('state'),'active_run':state.get('active_run'),
              'active_stage': (active_status or {}).get('state'),
              'active_step': (latest_train or active_status or {}).get('step'),
              'active_target_steps': (active_status or {}).get('target_steps'),
              'quality_gate':gate,'completed_students':len(results),'target_students':12,
              'independent_test_evaluation':False,'results':results}
    report_dir = ROOT / 'reports'; report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'latest_progress.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines = ['# 苹果随机异病例图文匹配 BCE / Pair-USA 进度', '',
             f"更新时间（UTC）：{report['updated_at']}", '',
             f"流水状态：{report['pipeline_state']}；已完成学生实验：{len(results)}/12。", '',
             f"当前组：{report['active_run'] or '尚无活动训练组'}。", '',
             f"当前阶段：{report['active_stage'] or '—'}；最新训练步：{report['active_step'] if report['active_step'] is not None else '—'}/{report['active_target_steps'] or '—'}。", '',
             '本轮使用其他病例的完整原始 caption 随机错排作为弱负例，未逐图审定语义矛盾。只汇总验证结果；未执行独立测试集模型评估。工程冒烟不计作正式实验。', '']
    if gate and not gate.get('ready_for_training'):
        lines += ['数据质量门槛尚未通过：' + '；'.join(gate.get('reasons',[])), '']
    lines += ['| 预算 | 方法 | 最佳步数 | 成对准确率 | AUROC | AP |',
              '|---|---|---:|---:|---:|---:|']
    for item in results:
        val = item['validation']
        def metric(key):
            value = val.get(key)
            return f'{value:.4f}' if isinstance(value,(float,int)) else '—'
        budget = str(int(item['budget'])) + '%' if item.get('budget') else '—'
        lines.append(f"| {budget} | {item['method']} | {item['best_step']} | {metric('paired_accuracy')} | {metric('auroc')} | {metric('average_precision')} |")
    lines += ['', '说明：这是单种子、固定步数的预算对照。训练配对已通过来源与结构检查，但不等于人工审定的语义负例。随机错配结果不能直接与局部病征 hard-negative 结果比较。', '']
    (report_dir/'最新实验进度.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('quality_gate','results')},ensure_ascii=False))

if __name__ == '__main__':
    main()
