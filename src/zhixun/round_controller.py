"""One real Qwen decision and continuation from explicit parent/data/history files.

This CLI is an explicit integration surface; platform invocation must be bound to it.
It neither runs an internal ten-round loop nor decides the platform's next parent.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import importlib

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
from baselines.data import read_dataset, json_value
from .checkpoint import load_checkpoint, kind, core, MAX_BYTES
from .experiment_agent.program import default_program
from .experiment_agent.task_tools import _validate_proposal
from .workspace_policy import decide
from .experiment_agent.unified_protocol import NUMERICAL_FUNCTIONS
from .experiment_agent.runtime import supervise


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def metrics(model, data):
    if model.meta['task']=='classification':
        from .steady_cli import evaluate
        return evaluate(model, data)
    error=model.predict(data['x'])-data['y']
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))))


def preflight(args):
    if not 1<=args.round<=10:raise ValueError('Round must be 1..10')
    if not 1<=args.cpu_seconds<=20:raise ValueError('Per-round CPU limit must be 1..20')
    if args.output.exists() or args.output.is_symlink():raise ValueError('Output must be new')
    parent_dir=args.output.resolve().parent
    if not parent_dir.is_dir() or not os.access(parent_dir,os.W_OK):raise ValueError('Output parent must exist and be writable')
    inputs={'parent':args.parent,'train':args.train,'validation':args.validation,'history':args.history}
    for name,path in inputs.items():
        if not path.is_file() or path.is_symlink():raise ValueError('Regular input file required: '+name)
        if args.output.resolve()==path.resolve() or args.output.resolve() in path.resolve().parents:raise ValueError('Output overlaps input')
    model=load_checkpoint(args.parent)
    if model.meta['profile']!=args.task:raise ValueError('Parent profile differs from task')
    train=read_dataset(args.train,model.meta['task'],model.meta['classes'],model.meta['paths'])
    valid=read_dataset(args.validation,model.meta['task'],model.meta['classes'],model.meta['paths'])
    if set(train['ids'])&set(valid['ids']):raise ValueError('Training and validation IDs overlap')
    if args.history.stat().st_size>1024*1024:raise ValueError('History exceeds 1 MiB')
    history=[json_value(line) for line in args.history.read_text().splitlines() if line.strip()]
    if len(history)!=args.round-1:raise ValueError('Provide every preceding platform round exactly once')
    for n,event in enumerate(history,1):
        if not isinstance(event,dict) or event.get('round_index')!=n:raise ValueError('History must contain ordered actual round_index records')
        if event.get('decision') not in ('promote','keep','retain'):raise ValueError('Actual platform decision required in history')
    return model,train,valid,history,{name:digest(path) for name,path in inputs.items()}


def run(args, workspace_io=None):
    model,train,valid,history,hashes=preflight(args)
    if args.check:
        return dict(valid=True,round_index=args.round,profile=args.task,policy='qwen',
                    device_metadata_required=False,device_split_verified=False,
                    model_calls=0,training_started=False)
    if workspace_io is None:
        workspace_io=importlib.import_module('workspace_io')
    if not callable(getattr(workspace_io,'model',None)):
        raise ValueError('Platform workspace_io.model callable required')
    root=args.output.resolve();root.mkdir();(root/'calls').mkdir();(root/'evidence').mkdir()
    shutil.copyfile(args.history,root/'evidence/platform-history.jsonl')
    write(root/'evidence/input-hashes.json',hashes)
    # This entry deliberately uses one metadata-independent learning contract in
    # every round, not a conditional substitute of fabricated device groups.
    reference=default_program(model.meta['task'],{'A':32,'B':64,'C':256}[args.task])
    reference=reference.replace('    counts = np.bincount(groups)\n    return len(y) / (len(counts) * counts[groups])','    return np.ones_like(y)')
    parent_core=core(model)
    context=dict(suggested_learning_rate={'A':.01,'B':.006,'C':.0005}[args.task],
                 current_learning_rate=model.meta['learning_rate'],parent_steps=model.meta['steps'],
                 parent_epochs=model.meta['epochs'],remaining_cpu_seconds=args.cpu_seconds,
                 device_groups_available=False)
    view=dict(profile=args.task,seed=42,next_round=args.round,required_next='experiment',
              actual_parent_sha256=hashes['parent'],parent_kind=kind(model),context=context,
              train_rows=len(train['y']),validation_rows=len(valid['y']),
              parent_training_metrics=metrics(model,train),parent_validation_metrics=metrics(model,valid),
              actual_platform_history=history,baseline_program=reference,
              numerical_functions=NUMERICAL_FUNCTIONS,device_groups=None,
              data_contract='id/inputs/answer only; meta excluded from learning and prediction',
              adoption_authority='external evaluator; supplied parent is authoritative')
    try:
        action,billing=decide(root/'calls/call-01',view,workspace_io)
        proposal=action['proposal'];_validate_proposal(proposal,args.task)
        case=root/'work';case.mkdir()
        shutil.copyfile(args.parent,case/'parent.npz')
        np.savez_compressed(case/'train.npz',x=train['x'],y=train['y'])
        write(case/'job.json',dict(proposal=proposal,parent_kind=kind(model),context=context,cpu_seconds=args.cpu_seconds))
        supervision=supervise(case,'zhixun.experiment_agent.task_worker')
        if not (case/'execution.json').exists():raise RuntimeError('Worker exited without execution receipt')
        execution=json_value((case/'execution.json').read_text())
        if supervision['exit_code']!=0 or execution['status']!='completed':raise RuntimeError('Continuation failed; see preserved worker evidence')
        candidate_path=case/'result/checkpoint.npz';candidate=load_checkpoint(candidate_path)
        child_core=core(candidate)
        if child_core.meta['steps']<=parent_core.meta['steps']:raise ValueError('No actual optimizer continuation')
        for name in ('mean','scale'):np.testing.assert_array_equal(getattr(parent_core,name),getattr(child_core,name))
        for name in ('profile','paths','classes','task','hidden','target_mean','target_scale','seed'):
            if child_core.meta[name]!=parent_core.meta[name]:raise ValueError('Inherited model contract changed')
        for name,path in [('parent',args.parent),('train',args.train),('validation',args.validation),('history',args.history)]:
            if digest(path)!=hashes[name]:raise ValueError('Input changed during the round')
        model_dir=root/'model';model_dir.mkdir();shutil.copyfile(candidate_path,model_dir/'checkpoint.npz')
        if sum(p.stat().st_size for p in model_dir.rglob('*') if p.is_file())>MAX_BYTES:raise ValueError('Model artifacts exceed 64 MiB')
        np.testing.assert_array_equal(candidate.predict(valid['x']),load_checkpoint(model_dir/'checkpoint.npz').predict(valid['x']))
        training=dict(execution['training'],round_index=args.round,parent_sha256=hashes['parent'],
                      checkpoint_sha256=digest(model_dir/'checkpoint.npz'),model_kind=kind(candidate),
                      actual_platform_history_sha256=hashes['history'],source_input_hashes=hashes,
                      policy='qwen',model_calls=1,model_usage=billing,
                      validation_before=view['parent_validation_metrics'],validation_after=metrics(candidate,valid),
                      adoption_decision=None,adoption_authority='external evaluator',
                      device_metadata_used=False,device_split_verified=False)
        write(root/'training.json',training)
        with (root/'history.jsonl').open('x') as stream:
            for event in training['history']:stream.write(json.dumps(event,allow_nan=False)+'\n')
        result=dict(status='candidate_ready',round_index=args.round,task=args.task,
                    checkpoint='model/checkpoint.npz',policy='qwen',model_calls=1,
                    parent_steps=parent_core.meta['steps'],candidate_steps=child_core.meta['steps'],
                    model_kind=kind(candidate),validation_metrics=training['validation_after'],
                    platform_verified=False)
        write(root/'result.json',result);return result
    except Exception as error:
        write(root/'result.json',dict(status='failed',error_type=type(error).__name__,
                                     fixed_rule_fallback=False,checkpoint_exported=False,
                                     evidence='calls/call-01 and work preserve available failure records'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=list('ABC'),required=True)
    parser.add_argument('--round',type=int,required=True)
    for name in ('parent','train','validation','history','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cpu-seconds',type=int,default=20)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    try:result=run(args)
    except Exception as error:
        print(json.dumps(dict(status='failed',error_type=type(error).__name__,fixed_rule_fallback=False)),file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
