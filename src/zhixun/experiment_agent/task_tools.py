"""Three-task experiment tools with editable MLP code and verified supplementary fits.

The executor owns data, parent state, score and acceptance. No model-service calls
occur here. This full-pack protocol is separate from the six-bearing study.
"""
from pathlib import Path
import argparse
import inspect
import hashlib
import json
import math
import shutil
import time
import numpy as np
from baselines.data import read_dataset
from baselines.model import Model
from .program import default_program, validate
from .runtime import supervise, evaluate
from .task_worker import load_checkpoint, core
from ..checkpoint import save_checkpoint, kind as checkpoint_kind, FORMAT, MAX_BYTES


def _write(path, value):
    pending=path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');pending.replace(path)


def _data(bundle, model, kind):
    path=Path(bundle)/'public'/f'{kind}.jsonl'
    data=read_dataset(path,model.meta['task'],model.meta['classes'],model.meta['paths'])
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    return data,rows


def _metrics(model, data, names):
    if model.meta['task']=='classification':
        from ..steady_cli import evaluate as classification_metrics
        return classification_metrics(model,data)
    return evaluate(model,data,names)


def _cached(root, split):
    with np.load(root/'data'/f'{split}.npz',allow_pickle=False) as d:values={k:d[k].copy() for k in d.files}
    values.setdefault('names',None)
    return values


def _state(root):
    state=json.loads((root/'state.json').read_text())
    if state['format']!='zhixun.task-tools.v2':raise ValueError('Use a new v2 run; preserved v1 runs are not migrated')
    return state


def inspect_bundle(bundle):
    """Validate the existing training-bundle format without writing or training."""
    bundle=Path(bundle).resolve()
    required=('checkpoint.npz','baseline-info.json','training.json','history.jsonl','public/train.jsonl','public/validation.jsonl')
    for name in required:
        if not (bundle/name).is_file():raise ValueError(f'Missing bundle file: {bundle/name}')
    if not isinstance(json.loads((bundle/'training.json').read_text()),dict):raise ValueError('Training report must be a JSON object')
    for line in (bundle/'history.jsonl').read_text().splitlines():
        if line.strip() and not isinstance(json.loads(line),dict):raise ValueError('History records must be JSON objects')
    model=load_checkpoint(bundle/'checkpoint.npz');profile=model.meta['profile'];task=model.meta['task']
    if profile not in ('A','B','C') or task!=('classification' if profile=='A' else 'regression'):raise ValueError('Task/profile mismatch')
    if json.loads((bundle/'baseline-info.json').read_text())['dataset_key']!=profile:raise ValueError('Wrong bundle metadata')
    train,rows=_data(bundle,model,'train');valid,vrows=_data(bundle,model,'validation')
    if set(train['ids'])&set(valid['ids']):raise ValueError('Train/validation overlap')
    if profile=='A':
        groups=train['y'];names=model.meta['classes'];train_names=np.array([names[y] for y in train['y']]);valid_names=np.array([names[y] for y in valid['y']])
    else:
        # Device IDs are optional provenance, never required training inputs.
        groups=None;names=[];train_names=None;valid_names=None
    return dict(model=model,profile=profile,task=task,train=train,valid=valid,
                groups=groups,names=names,train_names=train_names,valid_names=valid_names,
                train_rows=len(rows),validation_rows=len(vrows))


def prepare(bundle, output, cpu_budget):
    if type(cpu_budget) is not int or not 1<=cpu_budget<=600:raise ValueError('Specify an integer development CPU budget in 1..600 seconds')
    started=time.process_time();bundle,output=Path(bundle).resolve(),Path(output).resolve()
    if output==bundle or bundle in output.parents:raise ValueError('Output must be outside source data')
    checked=inspect_bundle(bundle)
    model,profile,task=(checked[k] for k in ('model','profile','task'))
    train,valid,groups,names,train_names,valid_names=(checked[k] for k in ('train','valid','groups','names','train_names','valid_names'))
    source=default_program(task,{'A':32,'B':64,'C':256}[profile]);initial=_metrics(model,valid,valid_names)
    output.mkdir(parents=True,exist_ok=False);(output/'data').mkdir()
    np.savez_compressed(output/'data/train.npz',x=train['x'],y=train['y'],**({} if groups is None else dict(groups=groups,names=train_names)))
    np.savez_compressed(output/'data/selection.npz',x=valid['x'],y=valid['y'],**({} if valid_names is None else dict(names=valid_names)))
    save_checkpoint(model,output/'data/initial.npz');(output/'baseline_program.py').write_text(source)
    state=dict(format='zhixun.task-tools.v2',profile=profile,task=task,selected='data/initial.npz',model_kind=checkpoint_kind(model),selected_trial=0,current_program=source,initial_metrics=initial,selected_metrics=initial,events=[],tool_cpu_seconds=time.process_time()-started,suggested_learning_rate={'A':.01,'B':.006,'C':.0005}[profile],max_attempts=10,max_tool_cpu_seconds=cpu_budget,group_names=names,device_metadata_used=False,device_split_verified=False,train_rows=checked['train_rows'],selection_rows=checked['validation_rows'],data_scope='supplied public training and selection only; device split not verified; no independent test',model_api_calls=0,platform_evaluations=0)
    evidence=output/'input-evidence';evidence.mkdir()
    hashes={}
    for name in ('checkpoint.npz','baseline-info.json','training.json','history.jsonl','public/train.jsonl','public/validation.jsonl'):
        hashes[name]=hashlib.sha256((bundle/name).read_bytes()).hexdigest()
    for name in ('baseline-info.json','training.json','history.jsonl'):
        shutil.copyfile(bundle/name,evidence/name)
    _write(evidence/'source-hashes.json',hashes)
    state['input_evidence']='input-evidence/source-hashes.json'
    _write(output/'state.json',state);return state


def observe(root):
    started=time.process_time();root=Path(root).resolve();state=_state(root);model=load_checkpoint(root/state['selected'],state['model_kind']);train=_cached(root,'train')
    from .worker import train as numerical_train
    operations=['core_code'];supplements=[];training_code=inspect.getsource(Model.forward)+'\n'+inspect.getsource(numerical_train)
    if state['profile']=='A':
        from ..steady_pair import SteadyPair
        training_code+='\n'+inspect.getsource(SteadyPair.fit_head)
        upstream=model.upstream if state['model_kind']=='a_readout' else model
        upstream_kind=model.upstream_kind if state['model_kind']=='a_readout' else state['model_kind']
        if upstream_kind=='core' or upstream.coef is None:supplements.append('steady_head')
    if state['profile']=='C':
        from ..boosted import ResidualBoost
        training_code+='\n'+inspect.getsource(ResidualBoost.fit);supplements.append('residual_trees')
    training_group_counts={name:int(sum(train['names']==name)) for name in state['group_names']}
    result=dict(profile=state['profile'],task=state['task'],parent_kind=state['model_kind'],parent_trial=state['selected_trial'],context=dict(suggested_learning_rate=state['suggested_learning_rate'],current_learning_rate=model.meta['learning_rate'],parent_epochs=model.meta['epochs'],parent_steps=model.meta['steps'],remaining_experiments=state['max_attempts']-len(state['events']),groups_available=state['task']=='classification'),available_operations=operations,available_supplements=supplements,training_group_counts=training_group_counts,model_state=dict(core_epochs=model.meta['epochs'],core_steps=model.meta['steps'],tree_iterations=model.iterations if state['model_kind']=='residual_boost' else 0,steady_head_fitted=state['profile']=='A' and upstream_kind=='steady_pair' and upstream.coef is not None),target_mean=model.meta['target_mean'],target_scale=model.meta['target_scale'],group_names=state['group_names'],train_rows=state['train_rows'],selection_rows=state['selection_rows'],current_metrics=state['selected_metrics'],current_training_metrics=_metrics(model,train,train['names']),history=state['events'],baseline_program=(root/'baseline_program.py').read_text(),current_program=state['current_program'],model_training_code=training_code,loss_contract='Raw class logits and integer labels; derivative with respect to logits' if state['task']=='classification' else 'Normalized scalar predictions/targets; derivative with respect to normalized predictions',data_scope=state['data_scope'])
    if state['profile']=='A':
        result['readout']=dict(names=['raw','x','hidden','logits','base_prob','classes'],
            class_order=model.meta['classes'],upstream_kind=upstream_kind,
            head_fitted=upstream_kind=='steady_pair' and upstream.coef is not None,
            expression=model.expression if state['model_kind']=='a_readout' else None,
            timing='Diagnostics describe selected parent; expression executes on same-trial continued upstream')
    state['tool_cpu_seconds']+=time.process_time()-started;_write(root/'state.json',state)
    result['context']['remaining_cpu_seconds']=state['max_tool_cpu_seconds']-state['tool_cpu_seconds'];return result


def _validate_proposal(proposal, profile=None):
    shared={'operation','hypothesis','expected_result','falsifier'}
    if not isinstance(proposal,dict) or not shared<=set(proposal):raise ValueError('Missing experiment fields')
    for k in shared:
        if not isinstance(proposal[k],str) or not 1<=len(proposal[k].strip())<=3000:raise ValueError('Nonempty bounded experiment text required')
    if proposal['operation']!='core_code':raise ValueError('Every experiment requires core_code with an explicit program')
    required=shared|{'program'}
    allowed=required|({'supplement','readout_expression'} if profile=='A' else {'supplement'} if profile=='C' else set())
    if not required<=set(proposal) or not set(proposal)<=allowed:raise ValueError('Unsupported joint proposal fields')
    validate(proposal['program'])
    if 'readout_expression' in proposal:
        from ..readout_expression import Expression
        Expression(proposal['readout_expression'])
    if 'supplement' in proposal:
        supplement=proposal['supplement']
        if not isinstance(supplement,dict):raise ValueError('supplement must be an explicit object')
        if profile=='A':
            if supplement!={'kind':'steady_head'}:raise ValueError('A supplement is exactly kind=steady_head')
        elif profile=='C':
            if set(supplement)!={'kind','trees','learning_rate'} or supplement['kind']!='residual_trees':raise ValueError('C supplement requires kind,trees,learning_rate')
            if type(supplement['trees']) is not int or not 1<=supplement['trees']<=80:raise ValueError('Expected 1..80 residual trees')
            rate=supplement['learning_rate']
            if type(rate) not in (int,float) or not math.isfinite(rate) or not 1e-6<=rate<=1:raise ValueError('Invalid tree learning rate')
        else:raise ValueError('Supplement is unavailable for this profile')


def experiment(root, proposal):
    started=time.process_time();root=Path(root).resolve();state=_state(root);n=len(state['events'])+1;remaining=state['max_tool_cpu_seconds']-state['tool_cpu_seconds']
    if n>state['max_attempts'] or remaining<1:raise ValueError('Experiment budget exhausted')
    parent=load_checkpoint(root/state['selected'],state['model_kind']);case=root/f'trial-{n:02d}'
    if case.exists():raise ValueError('Attempt directory exists; inspect preserved outcome before proceeding')
    case.mkdir();_write(case/'proposal.json',proposal)
    context=dict(suggested_learning_rate=state['suggested_learning_rate'],current_learning_rate=parent.meta['learning_rate'],parent_epochs=parent.meta['epochs'],parent_steps=parent.meta['steps'],remaining_experiments=state['max_attempts']-len(state['events']),remaining_cpu_seconds=remaining)
    _write(case/'job.json',dict(proposal=proposal,parent_kind=state['model_kind'],context=context,cpu_seconds=min(20,int(remaining))))
    try:_validate_proposal(proposal,state['profile'])
    except (ValueError,SyntaxError) as error:result=dict(status='preflight_failed',error=dict(type=type(error).__name__,message=str(error)),cpu_seconds=0.,wall_seconds=0.)
    else:
        shutil.copyfile(root/state['selected'],case/'parent.npz')
        selected_hashes={Path(state['selected']).name:hashlib.sha256((root/state['selected']).read_bytes()).hexdigest()}
        copied_hashes={'parent.npz':hashlib.sha256((case/'parent.npz').read_bytes()).hexdigest()}
        matches=list(selected_hashes.values())==list(copied_hashes.values())
        _write(case/'parent-provenance.json',dict(selected=state['selected'],selected_trial=state['selected_trial'],
            selected_files=selected_hashes,copied_files=copied_hashes,match=matches))
        if not matches:raise ValueError('Copied parent differs from selected checkpoint')
        shutil.copyfile(root/'data/train.npz',case/'train.npz')
        supervisor=supervise(case,'zhixun.experiment_agent.task_worker')
        result=json.loads((case/'execution.json').read_text()) if (case/'execution.json').exists() else dict(status='resource_terminated',error=dict(type='ResourceTermination',message='worker terminated before receipt'))
        result.update(supervisor)
    metric=None;accepted=False
    if result['status']=='completed':
        try:
            report=result['training'];candidate=load_checkpoint(case/'result'/report['checkpoint'],report['model_kind']);old,new=core(parent,state['model_kind']),core(candidate,report['model_kind'])
            for name in ('mean','scale'):np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
            for name in ('profile','paths','classes','task','hidden','target_mean','target_scale','seed'):
                if old.meta[name]!=new.meta[name]:raise ValueError('Immutable checkpoint metadata changed')
            if proposal['operation']=='core_code' and new.meta['steps']<=old.meta['steps']:raise ValueError('Core update failed to continue optimizer')
            valid=_cached(root,'selection');metric=_metrics(candidate,valid,valid['names']);key='macro_f1' if state['task']=='classification' else 'rmse'
            if not math.isfinite(metric[key]):raise ValueError('Nonfinite fixed evaluation')
            accepted=metric[key]>state['selected_metrics'][key] if key=='macro_f1' else metric[key]<state['selected_metrics'][key]
            np.save(case/'selection-predictions.npy',candidate.predict(valid['x']))
        except Exception as error:
            result.update(status='evaluation_failed',error=dict(type=type(error).__name__,message=str(error)[:1000]))
            metric=None;accepted=False
    event=dict(trial=n,parent_trial=state['selected_trial'],parent_kind=state['model_kind'],proposal=proposal,execution_status=result['status'],error=result.get('error'),child_cpu_seconds=result['cpu_seconds'],candidate_metrics=metric,incumbent_metrics_before=state['selected_metrics'],accepted=accepted)
    if accepted:
        state['selected']=str((case/'result'/result['training']['checkpoint']).relative_to(root));state['model_kind']=result['training']['model_kind'];state['selected_trial']=n;state['selected_metrics']=metric
        if proposal['operation']=='core_code':state['current_program']=proposal['program']
    event.update(selected_trial=state['selected_trial'],selected_kind=state['model_kind'],selected_metrics=state['selected_metrics'])
    state['events'].append(event);event['local_cpu_seconds']=time.process_time()-started;state['tool_cpu_seconds']+=result['cpu_seconds']+event['local_cpu_seconds'];state['complete']=len(state['events'])==state['max_attempts']
    _write(case/'execution.json',result);_write(case/'outcome.json',event);_write(root/'state.json',state);return event


def export_models(runs, output):
    """Export the exact selected functions as complete native checkpoints."""
    from ..solution import CHECKPOINTS
    runs,output=Path(runs).resolve(),Path(output).resolve()
    if output==runs or runs in output.parents:raise ValueError('Export must be outside preserved runs')
    states={task:_state(runs/task) for task in ('A','B','C')}
    models={}
    for task,state in states.items():
        if not state.get('complete') or len(state['events'])!=10 or state['profile']!=task:raise ValueError('Three completed task runs required')
        model=load_checkpoint(runs/task/state['selected'],state['model_kind'])
        if task=='B' and state['model_kind']!='core':raise ValueError('Unexpected B checkpoint kind')
        models[task]=model
    output.mkdir(parents=True,exist_ok=False)
    tasks={}
    for task,model in models.items():
        target=output/CHECKPOINTS[task];target.parent.mkdir(parents=True,exist_ok=True);save_checkpoint(model,target)
        tasks[task]=dict(checkpoint=CHECKPOINTS[task],format=FORMAT,kind=checkpoint_kind(model),development_metrics=states[task]['selected_metrics'])
    manifest=dict(format='zhixun.model-set.v1',tasks=tasks,origin='selected three-task experiment-tool results; see preserved proposals for decision provenance',hidden_test_verified=False,platform_protocol_verified=False,model_api_calls=sum(s['model_api_calls'] for s in states.values()))
    _write(output/'models.json',manifest)
    _write(output/'selection.json',{task:dict(selected_trial=s['selected_trial'],selected_kind=s['model_kind'],attempts=len(s['events']),tool_cpu_seconds=s['tool_cpu_seconds']) for task,s in states.items()})
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>MAX_BYTES:raise ValueError('Exported model artifacts including metadata exceed 64MiB')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-budget',type=int,required=True)
    p=sub.add_parser('observe');p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('experiment');p.add_argument('--run',type=Path,required=True);p.add_argument('--proposal',type=Path,required=True)
    p=sub.add_parser('export');p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.bundle,args.output,args.cpu_budget)
    elif args.command=='observe':result=observe(args.run)
    elif args.command=='export':result=export_models(args.runs,args.output)
    else:result=experiment(args.run,json.loads(args.proposal.read_text()))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
