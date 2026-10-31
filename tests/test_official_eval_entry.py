"""CPU contracts for the official separate evaluation command; no GPU imports."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from baseline import evaluate_delivery as evaluation
from run_pipeline import proxy


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def read(path):return json.loads(path.read_text())


@pytest.fixture
def suite(monkeypatch,tmp_path):
    root=tmp_path/'repository';scenes={'dl3dv':{'scene':'dl-scene'},'hw':{'scene':'billboard'},'mipnerf360':{'scene':'counter'}}
    write(root/'baseline/scenes.json',scenes);profile=root/'profile.json';write(profile,dict(device='cuda:0',environment={},environments={'fastgs':'f','resplat':'r'},conda='conda'))
    monkeypatch.setattr(evaluation,'ROOT',root);monkeypatch.setattr(evaluation,'DEFAULT_PROFILE',profile)
    calls=[]
    class Runner:
        fail_scene=None;bad_render_scene=None;nan_scene=None
        def __init__(self,path):self.profile=read(path);self.stage_seconds={}
        def conda(self,runtime,*args):return [runtime,*map(str,args)]
        def run(self,name,args,logs):
            calls.append(dict(name=name,args=args,logs=str(logs),profile=copy.deepcopy(self.profile)))
            model=Path(args[args.index('-m')+1] if '-m' in args else args[args.index('--model_path')+1]);scene=model.parents[1].name
            if name=='official_eval_render':
                names=read(Path(args[args.index('-s')+1])/'split.json')['test']
                if scene==self.bad_render_scene:names=names[:-1]
                for kind in ('test_renders','test_gt'):
                    for n in names:
                        p=model/kind/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fake PNG')
                write(model/'rendered_views.json',dict(iteration=int(args[args.index('--iteration')+1]),backend='fastgs',test_views=names))
            if name=='official_eval_metrics':
                if scene==self.fail_scene:raise subprocess.CalledProcessError(7,args)
                names=read(model/'rendered_views.json')['test_views'];mean=dict(psnr=27.,ssim=.85,lpips=.17)
                if scene==self.nan_scene:mean['lpips']=float('nan')
                write(model/'metrics.json',dict(num_views=len(names),mean=mean,per_view=[dict(name=n,**mean) for n in names]))
            self.stage_seconds[name]=1.25
    monkeypatch.setattr(evaluation,'Runner',Runner)
    return root,scenes,profile,calls,Runner


def trained(out,b=40.125,budget=120.5):
    record=dict(completed_iterations=5000,stop_reason='iterations',total_seconds=35.)
    write(out/'results/module_b/train_record.json',record)
    ply=out/'results/module_b/point_cloud/iteration_5000/point_cloud.ply';ply.parent.mkdir(parents=True);ply.write_bytes(b'ply\ncompleted model')
    write(out/'work/split.json',dict(train=['train.png'],test=['one.png','two.png']))
    report=dict(status='trained',evaluation_status='pending',metrics=None,proxy_score=None,b_iterations=5000,b_seconds=b,budget_seconds=budget,
        internal_b_seconds=35.,reference_internal_seconds=90.25,stage_seconds={'module_b':b,'reference':budget,'resplat':12.},total_wall_seconds=260.875)
    write(out/'report.json',report);return report


def populate(suite):
    root,scenes,*_=suite
    return {g:(root/'baseline/results'/g/s['scene']) for g,s in scenes.items()}


def test_no_arguments_evaluates_all_groups_and_preserves_B(suite):
    root,scenes,profile,calls,Runner=suite;outputs=populate(suite);original={g:trained(p) for g,p in outputs.items()}
    record_SHA={g:hashlib.sha256((p/'results/module_b/train_record.json').read_bytes()).hexdigest() for g,p in outputs.items()}
    assert evaluation.main([])==0 and len(calls)==6
    assert [c['name'] for c in calls]==['official_eval_render','official_eval_metrics']*3
    assert all('reference_native' not in str(c['args']) and 'train_confirmed' not in str(c['args']) for c in calls)
    for group,out in outputs.items():
        report=read(out/'report.json');assert report['status']==report['evaluation_status']=='complete'
        for key in ('b_seconds','budget_seconds','internal_b_seconds','reference_internal_seconds','stage_seconds','total_wall_seconds'):assert report[key]==original[group][key]
        assert report['proxy_score']==proxy(27.,.85,.17,40.125,120.5)
        assert set(report['evaluation']['stage_seconds'])=={'render','metrics'}
        assert report['evaluation']['mode']=='separate_invocation' and report['evaluation']['continuous_end_to_end_seconds'] is None
        assert hashlib.sha256((out/'results/module_b/train_record.json').read_bytes()).hexdigest()==record_SHA[group]
    summary=read(root/'baseline/results/summary.json');assert summary['success'] and summary['expected_cases']==summary['completed_cases']==3
    assert summary['official_score_100'] is summary['continuous_end_to_end_seconds'] is None
    assert summary['aggregate']['mean_B_seconds']==40.125
    assert len(list(csv.DictReader((root/'baseline/results/summary.csv').open())))==3
    assert 'not a continuous E2E' in (root/'baseline/results/summary.md').read_text()


def test_missing_group_is_nonzero_and_other_cases_still_evaluate(suite):
    root,_,_,calls,_=suite;outputs=populate(suite)
    for g,p in outputs.items():
        if g!='hw':trained(p)
    assert evaluation.main([])==1 and len(calls)==4
    summary=read(root/'baseline/results/summary.json');assert summary['completed_cases']==2 and summary['missing_cases']==1
    assert not summary['success'] and summary['aggregate'] is None and not outputs['hw'].exists()


def test_failed_metrics_remains_failed_without_stale_success(suite):
    root,_,_,calls,Runner=suite;outputs=populate(suite)
    for p in outputs.values():trained(p)
    Runner.fail_scene='billboard';assert evaluation.main([])==1 and len(calls)==6
    report=read(outputs['hw']/'report.json');assert report['evaluation_status']=='failed' and report['metrics'] is report['proxy_score'] is None
    assert report['b_seconds']==40.125 and 'CalledProcessError' in report['evaluation']['error']
    summary=read(root/'baseline/results/summary.json');assert summary['failed_cases']==1 and summary['aggregate'] is None
    Runner.fail_scene=None;assert evaluation.main(['hw'])==0


def test_group_and_custom_output_root(suite,tmp_path):
    root,scenes,profile,calls,_=suite;output=tmp_path/'custom';trained(output/'hw'/scenes['hw']['scene'])
    assert evaluation.main(['hw','--output-root',str(output),'--profile',str(profile),'--device','cpu'])==0
    assert len(calls)==2 and calls[1]['args'][-1]=='cpu'
    assert read(output/'summary.json')['requested_groups']==['hw']


def test_single_output_device_visibility_and_separate_summary(suite,tmp_path):
    root,_,profile,calls,_=suite;out=tmp_path/'one';trained(out);p=read(profile);p['environment']['CUDA_VISIBLE_DEVICES']='3,7';write(profile,p)
    summary=tmp_path/'summaries'
    assert evaluation.main(['--output',str(out),'--output-root',str(summary),'--profile',str(profile),'--device','cuda:1'])==0
    assert len(calls)==2 and all(c['profile']['environment']['CUDA_VISIBLE_DEVICES']=='7' for c in calls)
    assert calls[1]['args'][-1]=='cuda:0' and read(profile)['environment']['CUDA_VISIBLE_DEVICES']=='3,7'
    assert read(summary/'summary.json')['scope']=='single_output'
    assert read(out/'report.json')['evaluation']['requested_device']=='cuda:1'


@pytest.mark.parametrize('fault',['render','nan','missing_model','incomplete_B'])
def test_invalid_outputs_fail_before_a_success_report(suite,fault):
    root,_,_,calls,Runner=suite;out=populate(suite)['hw'];trained(out)
    if fault=='render':Runner.bad_render_scene='billboard'
    elif fault=='nan':Runner.nan_scene='billboard'
    elif fault=='missing_model':(out/'results/module_b/point_cloud/iteration_5000/point_cloud.ply').unlink()
    else:write(out/'results/module_b/train_record.json',dict(completed_iterations=4999,stop_reason='wall_clock'))
    assert evaluation.main(['hw'])==1
    assert read(out/'report.json')['evaluation_status']=='failed'
    if fault in ('missing_model','incomplete_B'):assert calls==[]
    if fault=='render':assert len(calls)==1


def test_over_budget_keeps_metrics_but_has_no_valid_total(suite):
    root,_,_,calls,_=suite;out=populate(suite)['hw'];trained(out,b=121,budget=120)
    assert evaluation.main(['hw'])==1 and len(calls)==2
    report=read(out/'report.json');assert report['evaluation_status']=='complete' and report['status']=='invalid_budget'
    assert report['b_seconds']==121 and report['proxy_score']['total'] is None and not report['within_budget']
    assert read(root/'baseline/results/summary.json')['aggregate'] is None


@pytest.mark.parametrize('args',[['unknown'],['hw','--output','somewhere']])
def test_invalid_selection_is_an_argument_error(suite,args):
    with pytest.raises(SystemExit) as error:evaluation.main(args)
    assert error.value.code==2 and suite[3]==[]


def test_shell_wrapper_help_is_independent_of_current_directory(tmp_path):
    script=Path(__file__).resolve().parents[1]/'baseline/run_eval.sh'
    result=subprocess.run(['bash',str(script),'--help'],cwd=tmp_path,env={**__import__('os').environ,'PYTHON':sys.executable},capture_output=True,text=True,check=True)
    assert '--output-root' in result.stdout and 'GROUP' in result.stdout
