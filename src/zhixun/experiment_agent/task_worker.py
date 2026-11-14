"""Continue core, then apply an explicit optional supplement on train only."""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import resource
import sys
import time
import numpy as np
from baselines.model import Model
from .worker import train as train_core


from ..checkpoint import load_checkpoint, save_checkpoint


def core(model, kind):return model if kind=='core' else model.base


def unchanged_core(before, after):
    if before.meta!=after.meta:raise ValueError('Supplementary training changed core metadata')
    for name in ('params','first','second'):
        for key,value in getattr(before,name).items():np.testing.assert_array_equal(value,getattr(after,name)[key])
    for name in ('mean','scale'):np.testing.assert_array_equal(getattr(before,name),getattr(after,name))


def train(case):
    case=Path(case);job=json.loads((case/'job.json').read_text());proposal=job['proposal'];kind=job['parent_kind']
    model=load_checkpoint(case/'parent.npz',kind);base=core(model,kind);old=copy.deepcopy(base)
    from .task_tools import _validate_proposal
    from ..readout import AReadout
    _validate_proposal(proposal,base.meta['profile'])
    result=case/'result';result.mkdir(exist_ok=False);start=time.monotonic()
    digest=lambda text:hashlib.sha256(text.encode()).hexdigest() if text is not None else None
    before_expression=model.expression if kind=='a_readout' else None
    upstream=model.upstream if kind=='a_readout' else model
    upstream_kind=model.upstream_kind if kind=='a_readout' else kind
    head_before=copy.deepcopy(upstream) if upstream_kind=='steady_pair' else None
    forest_before=upstream.booster.dump_model()['tree_info'] if upstream_kind=='residual_boost' and upstream.booster is not None else []
    save_checkpoint(base,case/'core-parent.npz');unchanged_core(old,load_checkpoint(case/'core-parent.npz','core'))
    numerical=train_core(case/'core-parent.npz',case/'train.npz',proposal['program'],job['context'],case/'numerical')
    updated=load_checkpoint(case/'numerical/checkpoint.npz','core')
    if updated.meta['steps']<=old.meta['steps']:raise ValueError('Core updater did not advance optimizer steps')
    if kind=='core':model=updated
    else:model.base=updated
    history=[dict(kind='core_epoch',**h) for h in numerical['history']]
    report=dict(operation='core_code',supplement=proposal.get('supplement'),numerical=numerical,
        core_steps_before=old.meta['steps'],core_steps_after=updated.meta['steps'],
        parent_checkpoint_sha256=hashlib.sha256((case/'parent.npz').read_bytes()).hexdigest(),
        core_parent_sha256=hashlib.sha256((case/'core-parent.npz').read_bytes()).hexdigest(),
        core_parent_matches_selected=True,program_sha256=digest(proposal['program']),train_only=True,
        complete=False,stage='core_completed',history=history)
    def record():
        report.update(model_kind=kind,checkpoint='checkpoint.npz',training_seconds=time.monotonic()-start)
        (result/'training.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    record()  # A subsequent supplement failure must retain the real core-update receipt.
    try:
        supplement=proposal.get('supplement')
        if supplement is not None and supplement['kind']=='steady_head':
            from ..steady_pair import SteadyPair
            upstream=model.upstream if kind=='a_readout' else model
            upstream_kind=model.upstream_kind if kind=='a_readout' else kind
            if upstream_kind=='core':upstream=SteadyPair(upstream)
            if upstream.coef is not None:raise ValueError('Existing steady head must be retained; refitting is not supported')
            with np.load(case/'train.npz',allow_pickle=False) as data:head=upstream.fit_head(data['x'],data['y'])
            if kind=='a_readout':model.upstream=upstream;model.upstream_kind='steady_pair'
            else:model=upstream;kind='steady_pair'
            unchanged_core(updated,model.base);history.append(dict(kind='head_fit',**head));report['head']=head
        elif supplement is not None and supplement['kind']=='residual_trees':
            from ..boosted import ResidualBoost
            rate=supplement['learning_rate']
            if kind=='core':model=ResidualBoost(model,leaves=15,min_leaf=30,rate=rate);kind='residual_boost'
            else:model.rate=rate
            with np.load(case/'train.npz',allow_pickle=False) as data:
                counts=np.bincount(data['groups']);weights=len(data['y'])/(len(counts)*counts[data['groups']])
                target=(data['y']-model.base.predict(data['x']))/model.meta['target_scale']
                retained=np.zeros(len(target)) if model.booster is None else model.booster.predict(model.base.normalized(data['x']),num_threads=1)
                # Existing fit uses target + init_model: LightGBM subtracts retained forest
                # predictions in its gradients. This is the effective pre-append residual.
                residual=target-retained;np.save(result/'residual-before-append.npy',residual)
                report.update(tree_iterations_before=model.iterations,residual_parent='same-trial updated core plus retained forest',
                    residual_before_append_mse=float(np.average(residual**2,weights=weights)))
                record()
                trees=model.fit(data['x'],data['y'],supplement['trees'],weights)
            unchanged_core(updated,model.base)
            if model.booster.dump_model()['tree_info'][:len(forest_before)]!=forest_before:raise ValueError('Existing forest changed during append')
            report.update(tree_iterations_after=model.iterations,existing_forest_unchanged=True)
            history.extend(dict(kind='residual_tree',tree_iteration=trees['tree_iterations_before']+i+1,loss=loss) for i,loss in enumerate(trees['train_residual_mse']))
        if 'readout_expression' in proposal:
            model=AReadout(model.upstream if kind=='a_readout' else model,
                           model.upstream_kind if kind=='a_readout' else kind,proposal['readout_expression']);kind='a_readout'
        upstream=model.upstream if kind=='a_readout' else model
        upstream_kind=model.upstream_kind if kind=='a_readout' else kind
        if head_before is not None and head_before.coef is not None:
            for name in ('indices','pair','mean','scale','coef','intercept'):
                np.testing.assert_array_equal(getattr(head_before,name),getattr(upstream,name))
        if upstream_kind=='residual_boost':
            if upstream.booster is not None and upstream.booster.dump_model()['tree_info'][:len(forest_before)]!=forest_before:raise ValueError('Retained forest changed')
            report.update(tree_iterations_before=len(forest_before),tree_iterations_after=upstream.iterations,existing_forest_unchanged=True)
        if old.meta['profile']=='A':
            report.update(readout_updated='readout_expression' in proposal,expression_before_sha256=digest(before_expression),
                expression_after_sha256=digest(model.expression if kind=='a_readout' else None),
                existing_head_unchanged=True,head_present=upstream_kind=='steady_pair' and upstream.coef is not None)
        report['stage']='supplement_and_readout_completed';record()
        destination=result/report['checkpoint'];save_checkpoint(model,destination)
        reloaded=load_checkpoint(destination,kind)
        with np.load(case/'train.npz',allow_pickle=False) as data:
            checked=data['x'] if kind=='a_readout' else data['x'][:32]
            np.testing.assert_array_equal(model.predict(checked),reloaded.predict(checked))
        report.update(complete=True,stage='serialization_verified');record();return report
    except Exception as error:
        # Preserve the failed candidate, never install it as selected state.
        record();destination=result/report['checkpoint']
        if not destination.exists():save_checkpoint(model,destination)
        report.update(error=dict(type=type(error).__name__,message=str(error)[:1000]));record()
        raise

def main():
    case=Path(sys.argv[1]);job=json.loads((case/'job.json').read_text());allowed=sorted(os.sched_getaffinity(0))[:4];os.sched_setaffinity(0,set(allowed))
    seconds=int(job['cpu_seconds']);resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds+1));resource.setrlimit(resource.RLIMIT_AS,(8000000000,8000000000));resource.setrlimit(resource.RLIMIT_FSIZE,(100000000,100000000));started=time.monotonic()
    try:report=train(case);status='completed';error=None
    except Exception as e:status='failed';error=dict(type=type(e).__name__,message=str(e)[:1000]);report=None
    usage=resource.getrusage(resource.RUSAGE_SELF)
    outcome=dict(status=status,error=error,training=report,cpu_seconds=usage.ru_utime+usage.ru_stime,wall_seconds=time.monotonic()-started,max_rss_kib=usage.ru_maxrss,cpu_affinity=allowed)
    (case/'execution.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n');print(json.dumps(outcome),flush=True)
    if status!='completed':sys.exit(1)


if __name__=='__main__':main()
