"""Summaries of separate official evaluations, without GPU imports."""
import csv
import io
import json
from pathlib import Path
import statistics


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def write_summary(output_root,cases,scope,groups):
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    completed=[r for r in cases if r['evaluation_status']=='complete']
    valid=[r for r in completed if r.get('within_budget') is True]
    success=len(valid)==len(cases) and bool(cases)
    mean_metrics={k:statistics.mean(r['metrics'][k] for r in completed) for k in ('psnr','ssim','lpips')} if completed else None
    aggregate=None
    if success:
        aggregate=dict(mean_metrics=mean_metrics,
            mean_quality_proxy=statistics.mean(r['proxy_score']['quality'] for r in valid),
            mean_time_proxy=statistics.mean(r['proxy_score']['time'] for r in valid),
            mean_total_proxy=statistics.mean(r['proxy_score']['total'] for r in valid),
            mean_B_seconds=statistics.mean(r['b_seconds'] for r in valid))
    summary=dict(schema_version=1,scope=scope,requested_groups=groups,status='complete' if success else 'incomplete',success=success,
        expected_cases=len(cases),completed_cases=len(completed),valid_budget_cases=len(valid),
        failed_cases=sum(r['evaluation_status']=='failed' for r in cases),missing_cases=sum(r['evaluation_status']=='missing' for r in cases),
        cases=cases,aggregate=aggregate,completed_cases_mean_metrics=mean_metrics,
        official_score_100=None,continuous_end_to_end_seconds=None,
        timing_scope='Separate evaluation invocation; original complete B and native budget preserved. No continuous E2E inferred.')
    atomic_json(output_root/'summary.json',summary)
    columns=('group','scene','evaluation_status','psnr','ssim','lpips','quality_proxy','time_proxy','total_proxy','b_seconds','budget_seconds','within_budget','evaluation_seconds','error')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in cases:
        metrics=row.get('metrics') or {};score=row.get('proxy_score') or {}
        writer.writerow({k:row.get(k,'') for k in columns if k not in ('psnr','ssim','lpips','quality_proxy','time_proxy','total_proxy')} |
            {k:metrics.get(k,'') for k in ('psnr','ssim','lpips')} |
            {k+'_proxy':score.get(k,'') for k in ('quality','time','total')})
    (output_root/'summary.csv').write_text(stream.getvalue(),newline='')
    def fmt(value,places=4):return '—' if value is None else f'{value:.{places}f}'
    def cell(value):return str(value).replace('|','\\|').replace('\n',' ')
    lines=['# Official evaluation summary','',f"Status: {summary['status']} ({len(completed)}/{len(cases)} evaluated; {len(valid)} within budget).",'',
        '|Group / scene|Status|PSNR|SSIM|VGG-LPIPS|B seconds|Eval seconds|Proxy total|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in cases:
        m=row.get('metrics') or {};s=row.get('proxy_score') or {}
        lines.append('|'+ '|'.join((cell(str(row['group'])+'/'+str(row['scene'])),cell(row['evaluation_status']),fmt(m.get('psnr'),3),fmt(m.get('ssim')),fmt(m.get('lpips')),
            fmt(row.get('b_seconds'),3),fmt(row.get('evaluation_seconds'),3),fmt(s.get('total'))))+'|')
    errors=[row for row in cases if row.get('error')]
    if errors:
        lines+=['']+[f"- {cell(row['group'])}/{cell(row['scene'])}: {cell(row['error'])}" for row in errors]
    if aggregate:lines+=['',f"Mean proxy: {aggregate['mean_total_proxy']:.4f}. Official score: unconfirmed."]
    lines+=['','Evaluation time is separate from preserved B time and native budget. Separate sessions are not a continuous E2E measurement.','']
    (output_root/'summary.md').write_text('\n'.join(lines));return summary
