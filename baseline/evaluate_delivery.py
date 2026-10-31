"""Official TEST evaluation for the default three groups or an existing single output."""
import argparse
import copy
import datetime
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_pipeline import DEFAULT_PROFILE,Runner,read,proxy
from baseline.evaluation_summary import atomic_json,write_summary


def number(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or positive and value<=0:
        raise ValueError(f'{name} must be finite and '+('positive' if positive else 'nonnegative'))
    return value


def device_profile(runner,requested):
    device=requested or runner.profile.get('device','cuda:0')
    if device=='cpu':return device,device
    match=re.fullmatch(r'cuda(?::(\d+))?',device)
    if not match:raise ValueError('--device must be cpu or cuda[:index]')
    index=int(match.group(1) or 0);environment=runner.profile.setdefault('environment',{})
    visible=environment.get('CUDA_VISIBLE_DEVICES',os.environ.get('CUDA_VISIBLE_DEVICES'))
    if visible is None:selected=str(index)
    else:
        choices=[v.strip() for v in str(visible).split(',') if v.strip()]
        if index>=len(choices):raise ValueError(f'{device} is outside CUDA_VISIBLE_DEVICES={visible}')
        selected=choices[index]
    environment['CUDA_VISIBLE_DEVICES']=selected
    return device,'cuda:0'


def preflight(out,report):
    iterations=report.get('b_iterations')
    if isinstance(iterations,bool) or not isinstance(iterations,int) or iterations<=0:raise ValueError('report.b_iterations must identify a completed model')
    number(report.get('b_seconds'),'report.b_seconds');number(report.get('budget_seconds'),'report.budget_seconds',positive=True)
    work=out/'work';model=out/'results/module_b'
    for path in (work/'split.json',model/'train_record.json',model/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'):
        if not path.is_file() or path.stat().st_size==0:raise FileNotFoundError(f'Missing or empty trained output: {path}')
    record=read(model/'train_record.json')
    if record.get('completed_iterations')!=iterations or record.get('stop_reason')!='iterations':raise ValueError('Module B is incomplete; evaluation cannot mark it successful')
    names=read(work/'split.json').get('test')
    if not isinstance(names,list) or not names or len(set(names))!=len(names):raise ValueError('work/split.json must contain a nonempty unique TEST list')
    if any(not isinstance(n,str) or Path(n).name!=n for n in names):raise ValueError('Unsafe TEST image name')
    return work,model,iterations,set(names)


def checked_render(model,expected,iterations):
    rendered=read(model/'rendered_views.json');names=rendered.get('test_views',[])
    if rendered.get('backend')!='fastgs' or rendered.get('iteration')!=iterations or len(names)!=len(expected) or set(names)!=expected:
        raise ValueError('Official renderer did not produce the complete requested TEST set')
    for name in ('test_renders','test_gt'):
        actual={p.name for p in (model/name).glob('*.png') if p.is_file()}
        if actual!=expected:raise ValueError(f'{name} file set differs from TEST split')


def checked_metrics(model,expected):
    data=read(model/'metrics.json');mean=data.get('mean',{});views=data.get('per_view',[])
    if data.get('num_views')!=len(expected) or len(views)!=len(expected) or {r.get('name') for r in views}!=expected:raise ValueError('Metrics do not cover every TEST view')
    for name in ('psnr','ssim','lpips'):
        value=mean.get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError(f'Invalid metric: {name}')
    return {k:mean[k] for k in ('psnr','ssim','lpips')}


def evaluate_case(group,scene,out,profile,device):
    out=out.resolve();report_path=out/'report.json';row=dict(group=group,scene=scene,output=str(out),evaluation_status='missing',metrics=None,proxy_score=None,
        b_seconds=None,budget_seconds=None,within_budget=None,evaluation_seconds=0.,error=None)
    if not report_path.is_file():row['error']=f'Missing training report: {report_path}';return row
    report=None;started=time.perf_counter();stages={};attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    try:
        report=read(report_path)
        if not isinstance(report,dict):raise ValueError('Training report must be an object')
        row.update(b_seconds=report.get('b_seconds'),budget_seconds=report.get('budget_seconds'))
        work,model,iterations,expected=preflight(out,report)
        row.update(b_seconds=report['b_seconds'],budget_seconds=report['budget_seconds'])
        runner=Runner(profile);requested,effective=device_profile(runner,device)
        preserved={k:copy.deepcopy(report.get(k)) for k in ('b_seconds','budget_seconds','internal_b_seconds','reference_internal_seconds','stage_seconds','total_wall_seconds')}
        report.update(status='evaluating',evaluation_status='running',metrics=None,proxy_score=None)
        report['evaluation']=dict(status='running',attempt=attempt,mode='separate_invocation',requested_device=requested,metrics_device=effective,
            timing_scope='evaluation only; original training/B timing preserved; not a continuous E2E')
        atomic_json(report_path,report);logs=out/'logs/evaluation'/attempt
        for name,command in (
            ('render',runner.conda('fastgs',ROOT/'baseline/render_test.py','-s',work,'-m',model,'--backend','fastgs','--iteration',iterations,'--eval','-r','1','--sh_degree','3','--quiet')),
            ('metrics',runner.conda('resplat',ROOT/'baseline/evaluate.py','--model_path',model,'--device',effective))):
            began=time.perf_counter()
            try:runner.run('official_eval_'+name,command,logs)
            finally:stages[name]=time.perf_counter()-began
            if name=='render':checked_render(model,expected,iterations)
        mean=checked_metrics(model,expected);score=proxy(**mean,b_seconds=report['b_seconds'],budget=report['budget_seconds'])
        if not score['valid']:score['total']=None
        for name,value in preserved.items():
            if name in report:assert report[name]==value,'Training timing changed during evaluation'
        report.update(status='complete' if score['valid'] else 'invalid_budget',evaluation_status='complete',metrics=mean,proxy_score=score,within_budget=score['valid'])
        elapsed=time.perf_counter()-started
        report['evaluation'].update(status='complete',stage_seconds=stages,elapsed_seconds=elapsed,continuous_end_to_end_seconds=None,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metrics_path=str(model/'metrics.json'),logs=str(logs))
        atomic_json(report_path,report)
        row.update(evaluation_status='complete',metrics=mean,proxy_score=score,b_seconds=report['b_seconds'],budget_seconds=report['budget_seconds'],within_budget=score['valid'],evaluation_seconds=elapsed)
        if not score['valid']:row['error']='Existing complete B time exceeds its matched native budget'
    except (OSError,ValueError,KeyError,TypeError,AssertionError,RuntimeError,subprocess.SubprocessError) as error:
        row.update(evaluation_status='failed',error=f'{type(error).__name__}: {error}',evaluation_seconds=time.perf_counter()-started)
        if isinstance(report,dict):
            report.update(status='failed',evaluation_status='failed',metrics=None,proxy_score=None)
            report['evaluation']=dict(status='failed',attempt=attempt,mode='separate_invocation',error=row['error'],stage_seconds=stages,
                elapsed_seconds=row['evaluation_seconds'],continuous_end_to_end_seconds=None)
            atomic_json(report_path,report)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='*',metavar='GROUP')
    parser.add_argument('--output',type=Path,help='Evaluate one existing single-scene pipeline output')
    parser.add_argument('--output-root',type=Path,default=ROOT/'baseline/results',help='Suite outputs and summary destination')
    parser.add_argument('--profile',type=Path,default=DEFAULT_PROFILE)
    parser.add_argument('--device',help='cpu for metrics, or cuda[:index] within profile visibility for rendering and metrics')
    args=parser.parse_args(argv);scenes=read(ROOT/'baseline/scenes.json')
    if args.output and args.groups:parser.error('--output cannot be combined with group arguments')
    if any(g not in scenes for g in args.groups):parser.error('Unknown group; choose '+', '.join(scenes))
    output_root=args.output_root.resolve();groups=list(dict.fromkeys(args.groups)) or list(scenes)
    if args.output:targets=[('single',args.output.name,args.output)];scope='single_output';groups=[]
    else:targets=[(g,scenes[g]['scene'],output_root/g/scenes[g]['scene']) for g in groups];scope='selected_groups'
    cases=[]
    for group,scene,out in targets:
        row=evaluate_case(group,scene,out,args.profile,args.device);cases.append(row)
        print(f"{group}/{scene}: {row['evaluation_status']}"+(f" — {row['error']}" if row['error'] else ''),flush=True)
    summary=write_summary(output_root,cases,scope,groups)
    print(f"Evaluation {summary['status']}: {summary['completed_cases']}/{summary['expected_cases']} evaluated, "
          f"{summary['valid_budget_cases']}/{summary['expected_cases']} within budget; summary: {output_root/'summary.json'}",flush=True)
    return 0 if summary['success'] else 1


if __name__=='__main__':raise SystemExit(main())
