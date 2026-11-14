"""Run the same three-task experiment tools under a fixed rule or designated model."""
from pathlib import Path
import argparse,json,os,resource,sys,time
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',BASELINE_CPU_THREADS='1')
import numpy as np
from .task_tools import prepare,observe,experiment,export_models,inspect_bundle,_state,_write,_cached,_metrics
from .task_worker import load_checkpoint
from .driver import decide, validate_prices, FORMAT
from .unified_protocol import parse,NUMERICAL_FUNCTIONS,REFERENCE_RECIPES
from .evidence import changed_functions


def short_metrics(metrics):
    if metrics is None:return None
    keys=('macro_f1','accuracy') if 'macro_f1' in metrics else ('rmse','mae')
    out={k:round(metrics[k],6) for k in keys}
    if 'recall' in metrics:out['class_recall']={k:None if v['recall'] is None else round(v['recall'],6) for k,v in metrics['recall'].items()}
    if 'groups' in metrics:out['groups']={k:{f:round(v[f],6) for f in ('rmse','bias')} for k,v in metrics['groups'].items()}
    return out


def history(root,state):
    programs={0:(root/'baseline_program.py').read_text()};records=[]
    for event in state['events']:
        n=event['trial'];p=event['proposal'];parent_program=programs[event['parent_trial']];record={k:event[k] for k in ('trial','parent_trial','parent_kind','execution_status','accepted','selected_trial','selected_kind')}
        record.update(operation=p.get('operation'),hypothesis_unverified=str(p.get('hypothesis',''))[:300],hypothesis_abbreviated=len(str(p.get('hypothesis','')))>300,supplement=p.get('supplement'),candidate_metrics=short_metrics(event['candidate_metrics']),selected_metrics=short_metrics(event['selected_metrics']),error=event['error'])
        if p.get('operation')=='core_code':
            try:record['changed_functions']=changed_functions(parent_program,p['program'])
            except (ValueError,SyntaxError,KeyError):record['changed_functions']=None
            if state['profile']=='A':
                import hashlib
                expression=p.get('readout_expression')
                record.update(readout_updated='readout_expression' in p,
                    expression_sha256=hashlib.sha256(expression.encode()).hexdigest() if isinstance(expression,str) else None,
                    expression_length=len(expression) if isinstance(expression,str) else None)
        if (root/f'trial-{n:02d}/result/training.json').exists():
            report=json.loads((root/f'trial-{n:02d}/result/training.json').read_text());h=report['history'];record['actual_training']={k:v for k,v in report.items() if k in ('operation','model_kind','core_steps_before','core_steps_after','tree_iterations_before','tree_iterations_after')}
            record['actual_training'].update({k:report[k] for k in ('complete','readout_updated','expression_before_sha256','expression_after_sha256','existing_head_unchanged','existing_forest_unchanged','supplement','residual_parent','stage') if k in report})
            if 'numerical' in report:record['actual_training']['plan']=report['numerical']['plan']
            core_losses=[row['loss'] for row in h if row.get('kind')=='core_epoch' and 'loss' in row]
            tree_losses=[row['loss'] for row in h if row.get('kind')=='residual_tree' and 'loss' in row]
            if core_losses:record['actual_training'].update(core_loss_first=core_losses[0],core_loss_last=core_losses[-1],core_loss_scope='submitted core objective per epoch')
            if tree_losses:record['actual_training'].update(tree_loss_first=tree_losses[0],tree_loss_last=tree_losses[-1],tree_loss_scope='LightGBM training residual MSE in normalized target units')
            if 'head' in report:record['actual_training']['head_fit']=report['head']
        programs[n]=p['program'] if event['accepted'] and p.get('operation')=='core_code' else parent_program
        records.append(record)
    return records


def canonical_request(view):
    """Replace only exact duplicate read results with a top-level JSON Pointer.

    Build new feedback objects; the original tool result remains intact for the
    controller's tool-result.json receipt and any later provenance inspection.
    """
    mapping={'history':'history','current_program':'current_program',
             'baseline_program':'baseline_program','training_code':'model_training_code'}
    feedback=[]
    for item in view.get('current_tool_results',[]):
        action=item.get('request',{});key=None
        if action.get('tool')=='read':
            resource=action.get('resource')
            key=mapping.get(resource)
            if resource==f"trial/{view.get('next_round',0)-1}/proposal":key='last_proposal'
        if key in view and 'result' in item and 'result_ref' not in item:
            encode=lambda value:json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
            if encode(item['result'])==encode(view[key]):
                item={k:v for k,v in item.items() if k!='result'}
                item['result_ref']='/'+key
        feedback.append(item)
    return dict(view,current_tool_results=feedback)


def request(root,tool_feedback):
    view=observe(root);state=_state(root);view['history']=history(root,state)
    if state['events']:
        last=state['events'][-1];view['last_proposal']=last['proposal']
    view.update(seed=42,next_round=len(state['events'])+1,remaining_calls=21-len(list((root/'calls').glob('call-*'))),required_next='finish' if len(state['events'])==10 else 'experiment' if tool_feedback else 'one read/analyze or experiment',numerical_functions=NUMERICAL_FUNCTIONS,reference_recipe=REFERENCE_RECIPES[state['profile']],current_tool_results=tool_feedback)
    return canonical_request(view)


def inspect_request(root,action,view):
    if action['tool']=='read':
        name=action['resource'];mapping={'training_code':'model_training_code','current_program':'current_program','baseline_program':'baseline_program','history':'history'}
        if name in mapping:return view[mapping[name]]
        _,n,kind=name.split('/');n=int(n);state=_state(root)
        if n>len(state['events']):raise ValueError('Cannot read future experiment')
        if kind=='proposal':return state['events'][n-1]['proposal']
        return json.loads((root/f'trial-{n:02d}/execution.json').read_text())
    state=_state(root);model=load_checkpoint(root/state['selected'],state['model_kind']);data=_cached(root,action['dataset']);train=_cached(root,'train');pred=model.predict(data['x']);mode=action['view']
    if mode=='readout_summary':
        if state['profile']!='A':raise ValueError('readout_summary is A-only')
        upstream=model.upstream if state['model_kind']=='a_readout' else model
        base=upstream if (model.upstream_kind if state['model_kind']=='a_readout' else state['model_kind'])=='core' else upstream.base
        values=dict(logits=base.forward(base.normalized(data['x']))[1],base_prob=upstream.predict(data['x']),selected_prob=pred)
        def quantiles(array):
            return [None if not np.any(data['y']==i) else
                    [[float(f'{v:.6g}') for v in column] for column in np.quantile(array[data['y']==i],[.1,.5,.9],axis=0).T]
                    for i in range(6)]
        return dict(class_order=model.meta['classes'],class_counts=np.bincount(data['y'],minlength=6).tolist(),
            parent_trial=state['selected_trial'],core_steps=base.meta['steps'],dataset=action['dataset'],
            timing='selected parent before candidate training',axes=['true_class','output_class','quantile'],
            quantiles=[.1,.5,.9],summaries={k:quantiles(v) for k,v in values.items()})
    if mode=='group_errors':return _metrics(model,data,data['names'])
    if model.meta['task']=='classification':
        if mode=='target_bins':return dict(available=False,reason='RUL bins do not apply to class labels; use confusion/recall')
        error=(pred.argmax(1)!=data['y']).astype(float)
    else:error=pred-data['y']
    if mode=='target_bins':
        edges=np.unique(np.quantile(train['y'],[0,.25,.5,.75,1]));ids=np.digitize(data['y'],edges[1:-1])
        return dict(training_target_edges=edges.tolist(),bins=[dict(bin=int(i),n=int(sum(ids==i)),bias=float(error[ids==i].mean()),rmse=float(np.sqrt(np.mean(error[ids==i]**2)))) for i in sorted(set(ids))])
    base=model if state['model_kind']=='core' else model.base;x=base.normalized(data['x']);z=x-x.mean(0);e=error-error.mean();den=np.sqrt(np.sum(z*z,axis=0)*np.sum(e*e));cor=np.divide(z.T@e,den,out=np.zeros(x.shape[1]),where=den>0);indices=np.argsort(-abs(cor))[:8]
    return dict(association_only=True,error_definition='incorrect indicator' if state['task']=='classification' else 'prediction minus target in physical units',features=[dict(feature=model.meta['paths'][i],correlation=float(cor[i])) for i in indices])


def receipts(root):
    values=[];authorizations=0
    for path in sorted((root/'calls').glob('call-*')):
        file=path/'billing.json'
        bill=json.loads(file.read_text()) if file.exists() else dict(status='unknown',actual_bill=False)
        receipt=path/'transport/result.json'
        transport=json.loads(receipt.read_text()) if receipt.exists() else None
        count=transport['sampling_authorizations'] if transport is not None else bill.get('sampling_authorizations',0)
        authorizations+=count
        values.append(dict(bill,call=path.name,sampling_authorizations=count))
    return values,authorizations


def remaining_deadline(root):
    budget=json.loads((Path(root)/'runtime-budget.json').read_text())
    if budget['format']!=FORMAT or budget['task_wall_limit_seconds']!=1800 or budget['call_limit']!=21:
        raise ValueError('Only the new fixed-budget run format is accepted')
    if budget['expires_at_unix']-budget['created_at_unix']!=1800 or budget['deadline_monotonic']-budget['started_monotonic']!=1800:
        raise ValueError('Original deadline record is inconsistent')
    remaining=budget['expires_at_unix']-time.time()
    deadline=min(budget['deadline_monotonic'],time.monotonic()+remaining)
    if deadline<=time.monotonic():raise RuntimeError('Original absolute task deadline exhausted')
    return deadline


def rule_proposal(view,events):
    task=view['profile'];n=len(events)+1;shared=dict(hypothesis='Fixed explicit core continuation with optional supplement; joint A/C scores unverified',expected_result='Improve the fixed primary metric',falsifier='No strict primary improvement')
    supplement=None
    if task=='C':
        rate=view['context']['suggested_learning_rate']
        supplement=dict(kind='residual_trees',trees=80,learning_rate=.05*2**(-sum(not e['accepted'] for e in events)))
    elif task=='A':
        rate=.01*2**(-sum(not e['accepted'] for e in events))
        if n==10:supplement=dict(kind='steady_head')
    else:
        tried={float(e['proposal']['program'].split('def learning_rate(step, context):\n    return ')[1].split('\n')[0]) for e in events if e['parent_trial']==view['parent_trial']}
        rate=next(.006*2**i for i in [0,1,*range(-1,-14,-1)] if .006*2**i not in tried)
    proposal=dict(operation='core_code',program=view['baseline_program'].replace('return context["suggested_learning_rate"]','return '+repr(rate)),**shared)
    if supplement is not None:proposal['supplement']=supplement
    return proposal


def resume_checkpoint(root,policy,cpu_budget):
    """Explicit continuation only after a verified pre-sampling service failure."""
    root=Path(root);state=_state(root);summary=json.loads((root/'controller-summary.json').read_text())
    if policy!='qwen' or summary['policy']!=policy or summary['complete']:
        raise ValueError('Resume requires an interrupted model-driven task')
    if summary['cpu_budget']!=cpu_budget or state['max_tool_cpu_seconds']!=cpu_budget:
        raise ValueError('Resume cannot reset or change the CPU budget')
    if summary['experiments']!=len(state['events']) or summary['selected_trial']!=state['selected_trial'] or summary['final_metrics']!=state['selected_metrics']:
        raise ValueError('Saved summary and selected state disagree')
    cases=sorted(root.glob('trial-*'))
    if [c.name for c in cases]!=[f'trial-{i:02d}' for i in range(1,len(state['events'])+1)]:
        raise ValueError('Unrecorded experiment requires investigation before continuing')
    calls=sorted((root/'calls').glob('call-*'))
    if not calls or [c.name for c in calls]!=[f'call-{i:02d}' for i in range(1,len(calls)+1)]:
        raise ValueError('Call sequence is incomplete')
    if summary.get('runtime_format')!=FORMAT or state.get('runtime_format')!=FORMAT:
        raise ValueError('Historical run formats cannot resume in this copy')
    remaining_deadline(root)
    if summary['task_deadline_unix']!=json.loads((root/'runtime-budget.json').read_text())['expires_at_unix']:
        raise ValueError('Summary deadline differs from original budget')
    bills,attempts=receipts(root)
    if any(b['status']=='unknown' for b in bills) or attempts!=state['model_api_calls'] or attempts!=summary['model_api_calls']:
        raise ValueError('Unresolved usage cannot be retried')
    if len(calls)!=state['model_call_attempts'] or len(calls)!=summary['model_call_attempts'] or len(calls)>=21:
        raise ValueError('Original attempt budget exhausted or inconsistent')
    for call in calls:
        manifest=json.loads((call/'manifest.json').read_text())
        if manifest['format']!=FORMAT:raise ValueError('Unrecognized call provenance')
        if set(manifest['files'])!={'request.json','price.json','prompt.txt'}:
            raise ValueError('Incomplete input provenance')
        import hashlib
        for name,h in manifest['files'].items():
            if name not in ('request.json','price.json','prompt.txt') or hashlib.sha256((call/name).read_bytes()).hexdigest()!=h:
                raise ValueError('Call inputs changed')
    last=calls[-1];result=json.loads((last/'result.json').read_text());transport=json.loads((last/'transport/result.json').read_text())
    events=[json.loads(line) for line in (last/'transport/protocol.jsonl').read_text().splitlines()]
    if result['format']!=FORMAT or result['status']!='failed' or result['transport']!=transport:
        raise ValueError('Missing matching terminal failure receipts')
    if bills[-1]['status']!='no_sampling_request' or transport['status']!='worker_error' or transport['failed_phase']!='initialization' or transport['sampling_authorizations']!=0 or transport['sampling_started_markers']!=0 or transport['response_received']:
        raise ValueError('Only a proven pre-sampling worker failure may resume')
    if not events or events[-1].get('event')!='error' or events[-1].get('phase')!='initialization' or any(e.get('event') in ('sampling_started','result','ready') for e in events):
        raise ValueError('Protocol does not prove a pre-sampling failure')
    if any((last/name).exists() for name in ['response.json','provider-usage.json','decision.json']):
        raise ValueError('Never replay a potentially paid call')
    pending=json.loads((last/'request.json').read_text())
    if pending['next_round']!=len(state['events'])+1 or pending['parent_trial']!=state['selected_trial']:
        raise ValueError('Pending request does not match the selected parent')
    if state['tool_cpu_seconds']>=cpu_budget:
        raise ValueError('Original CPU budget already exhausted')

    return summary,pending.get('current_tool_results',[])


def run_task(bundle,root,policy,cpu_budget,prices,*,resume=False):
    root=Path(root);started=time.monotonic();resume_count=0
    if policy=='qwen':validate_prices(prices)
    if resume:
        cpu=time.process_time();previous,feedback=resume_checkpoint(root,policy,cpu_budget)
        archive=root/'continuations';archive.mkdir(exist_ok=True);resume_count=len(list(archive.glob('*.json')))+1
        _write(archive/f'{resume_count:02d}.json',dict(previous_summary=previous,state_before=_state(root),previous_failed_call=sorted((root/'calls').glob('call-*'))[-1].name,reason='explicit continuation after pre-sampling service failure; no paid response replayed',paused_wall_time_excluded=False))
        state=_state(root);state['tool_cpu_seconds']+=time.process_time()-cpu;_write(root/'state.json',state)
        task_deadline=remaining_deadline(root)
    else:
        prepare(bundle,root,cpu_budget);(root/'calls').mkdir();started=time.monotonic();feedback=[]
        now=time.time();task_deadline=started+1800
        _write(root/'runtime-budget.json',dict(format=FORMAT,task_wall_limit_seconds=1800,call_limit=21,
            created_at_unix=now,expires_at_unix=now+1800,started_monotonic=started,deadline_monotonic=task_deadline))
        state=_state(root);state.update(runtime_format=FORMAT,model_call_attempts=0);_write(root/'state.json',state)
    budget_record=json.loads((root/'runtime-budget.json').read_text())
    final_review=None;success=False
    try:
        while True:
            state=_state(root)
            if policy=='rule' and len(state['events'])==10:success=True;break
            if state['tool_cpu_seconds']>=cpu_budget or time.monotonic()>=task_deadline:raise RuntimeError('Run resource budget exhausted')
            if policy=='qwen' and len(list((root/'calls').glob('call-*')))>=21:raise RuntimeError('Model call budget exhausted before final review')
            view=request(root,feedback)
            if policy=='rule':action=dict(tool='experiment',reflection='Fixed rule control',proposal=rule_proposal(view,state['events']))
            else:
                directory=root/'calls'/f"call-{len(list((root/'calls').glob('call-*')))+1:02d}";cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
                try:action,usage=decide(directory,view,python=sys.executable,prices=prices,task_deadline=task_deadline,worker_module='zhixun.unified_policy_worker',decision_parser=parse)
                finally:
                    after=resource.getrusage(resource.RUSAGE_CHILDREN);state=_state(root);state['tool_cpu_seconds']+=time.process_time()-cpu+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
                    bills,attempts=receipts(root);state['model_api_calls']=attempts;state['model_call_attempts']=len(bills);_write(root/'state.json',state);_write(root/'calls.json',bills)
            if time.monotonic()>=task_deadline:raise RuntimeError('Task deadline crossed before action execution')
            if action['tool']=='finish':
                if len(_state(root)['events'])!=10:raise ValueError('Final review before ten attempts')
                final_review=action;success=True;break
            if len(_state(root)['events'])==10:raise ValueError('Only final review is allowed after ten attempts')
            if action['tool'] in ('read','analyze'):
                if feedback:raise ValueError('Only one inspection per experiment')
                cpu=time.process_time();result=inspect_request(root,action,view);state=_state(root);state['tool_cpu_seconds']+=time.process_time()-cpu;_write(root/'state.json',state);feedback=[dict(request=action,result=result)];_write(directory/'tool-result.json',feedback)
                if time.monotonic()>=task_deadline:raise RuntimeError('Task deadline crossed during inspection')
                continue
            event=experiment(root,action['proposal']);_write(root/f"trial-{event['trial']:02d}/reflection.json",dict(reflection=action['reflection'],policy=policy));feedback=[]
            if time.monotonic()>=task_deadline:raise RuntimeError('Task deadline crossed during CPU tool; no next phase')
            print(json.dumps(dict(experiment_completed=event['trial'],task=state['profile'],policy=policy,status=event['execution_status'],accepted=event['accepted'],selected_metrics=short_metrics(event['selected_metrics']))),flush=True)
    finally:
        state=_state(root);bills,attempts=receipts(root);known=[b for b in bills if b['status']=='usage_returned'];summary=dict(complete=success,policy=policy,profile=state['profile'],experiments=len(state['events']),initial_metrics=state['initial_metrics'],final_metrics=state['selected_metrics'],selected_trial=state['selected_trial'],model_kind=state['model_kind'],cpu_budget=cpu_budget,measured_cpu_seconds=state['tool_cpu_seconds'],wall_seconds=max(0,time.time()-budget_record['created_at_unix']),runtime_format=FORMAT,task_deadline_unix=budget_record['expires_at_unix'],model_call_attempts=len(bills),model_api_calls=attempts,model_call_count_scope='local sampling authorizations, not verified HTTP dispatches',input_tokens=sum(b['input_tokens'] for b in known),output_tokens=sum(b['output_tokens'] for b in known),estimated_known_cny=sum(b['estimated_uncached_cny'] for b in known),unknown_billing_calls=[b['call'] for b in bills if b['status']=='unknown'],actual_bill_verified=False,final_review=final_review,continuation_count=resume_count,wall_scope="original absolute deadline including resume gaps; preparation excluded",platform_evaluations=0)
        _write(root/'controller-summary.json',summary)
    return summary


def inspect_resume(args, workspace):
    """Read-only validation of the exact saved run before any resumed work."""
    if args.policy!='qwen':raise ValueError('Resume requires the explicitly selected qwen policy')
    runs=workspace/'runs'
    if not runs.is_dir() or (workspace/'summary.json').exists():
        raise ValueError('Resume requires an incomplete workspace with runs/')
    if (workspace/'models').exists() or (workspace/'models').is_symlink():
        raise ValueError('Resume model output already exists; preserve it and investigate')
    for directory in (workspace,runs):
        if not os.access(directory,os.W_OK|os.X_OK):raise ValueError('Resume workspace must be writable')
    results={};pending=set()
    expected_files={'checkpoint.npz','baseline-info.json','training.json','history.jsonl','public/train.jsonl','public/validation.jsonl'}
    import hashlib
    for task in 'ABC':
        root=runs/task
        if not root.exists():continue
        for directory in (root,root/'calls'):
            if not directory.is_dir() or not os.access(directory,os.W_OK|os.X_OK):raise ValueError('Resume task directories must be writable')
        for name in ('state.json','controller-summary.json'):
            if not os.access(root/name,os.W_OK):raise ValueError('Resume state files must be writable')
        supplied=json.loads((root/'input-evidence/source-hashes.json').read_text())
        if set(supplied)!=expected_files:raise ValueError('Incomplete original input binding')
        for name,digest in supplied.items():
            if hashlib.sha256((args.bundles/task/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Resume input changed: {task}/{name}')
        summary=json.loads((root/'controller-summary.json').read_text());state=_state(root)
        if summary.get('runtime_format')!=FORMAT or state.get('runtime_format')!=FORMAT:
            raise ValueError('Resume accepts only this runtime format')
        if summary['profile']!=task or summary['policy']!=args.policy or summary['cpu_budget']!=args.cpu_budget or len(state['events'])!=summary['experiments'] or state['selected_metrics']!=summary['final_metrics']:
            raise ValueError('Existing task does not match requested run')
        if summary['complete']:
            if len(state['events'])!=10 or not summary['final_review']:raise ValueError('Completed task has missing experiments or review')
            results[task]=summary
        else:
            resume_checkpoint(root,args.policy,args.cpu_budget);pending.add(task)
    existing=[task for task in 'ABC' if (runs/task).exists()]
    if len(pending)!=1 or existing!=list('ABC'[:len(existing)]) or next(iter(pending))!=existing[-1]:
        raise ValueError('Expected one interrupted task after a completed task prefix')
    return results,pending


def preflight(args):
    """Reject invalid launch inputs before creating outputs or sampling."""
    if not 1<=args.cpu_budget<=600:raise ValueError('--cpu-budget must be in 1..600 seconds')
    bundles=args.bundles.resolve();workspace=args.output.resolve()
    if workspace==bundles or bundles in workspace.parents or workspace in bundles.parents:
        raise ValueError('Input bundles and output workspace must not overlap')
    if not args.resume and workspace.exists():raise ValueError('Output workspace must be new')
    resume_plan=inspect_resume(args,workspace) if args.resume else ({},set())
    if not args.resume and (not workspace.parent.is_dir() or not os.access(workspace.parent,os.W_OK|os.X_OK)):
        raise ValueError('Output parent must exist and be writable')
    prices=None
    if args.policy=='qwen':
        if not args.price_record or not args.tokenizer or not os.environ.get('PYTRIO_API_KEY'):
            raise ValueError('Explicit --price-record, --tokenizer and runtime PYTRIO_API_KEY required')
        prices=json.loads(args.price_record.read_text());validate_prices(prices)
        previous=os.environ.get('PYTRIO_TOKENIZER_PATH')
        try:
            os.environ['PYTRIO_TOKENIZER_PATH']=str(args.tokenizer.resolve())
            from ..policy_worker import load_tokenizer
            load_tokenizer()  # local-only provenance and deserialization; no service call
        finally:
            if previous is None:os.environ.pop('PYTRIO_TOKENIZER_PATH',None)
            else:os.environ['PYTRIO_TOKENIZER_PATH']=previous
    tasks={}
    for task in 'ABC':
        data=inspect_bundle(bundles/task)
        if data['profile']!=task:raise ValueError(f'Bundle directory {task} has the wrong profile')
        tasks[task]=dict(train_rows=data['train_rows'],validation_rows=data['validation_rows'])
    return prices,dict(valid=True,policy=args.policy,bundles=str(bundles),output=str(workspace),
                       tasks=tasks,model_api_calls=0,training_started=False,
                       platform_protocol_verified=False),resume_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=['rule','qwen'],required=True)
    p.add_argument('--cpu-budget',type=int,required=True)
    p.add_argument('--price-record',type=Path)
    p.add_argument('--tokenizer',type=Path)
    p.add_argument('--check',action='store_true',help='Validate all launch inputs without writes, training or model calls')
    p.add_argument('--resume',action='store_true',help='Continue a verified pre-sampling failure in the same workspace')
    a=p.parse_args()
    try:
        prices,checked,resume_plan=preflight(a)
    except Exception as error:
        message=str(error)
        key=os.environ.get('PYTRIO_API_KEY')
        if key:message=message.replace(key,'[REDACTED]')
        print(json.dumps(dict(stage='preflight',error_type=type(error).__name__,message=message)),file=sys.stderr)
        raise SystemExit(2)
    if a.check:
        print(json.dumps(checked));return
    if sys.platform!='linux':raise RuntimeError('Use the native Linux host')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_AS,(8000000000,8000000000))
    os.environ.pop('SWANLAB_API_KEY',None)
    if a.policy=='qwen':os.environ['PYTRIO_TOKENIZER_PATH']=str(a.tokenizer.resolve())
    workspace=a.output.resolve();output=workspace/'runs';results={};resume_tasks=set()
    if a.resume:
        results,resume_tasks=resume_plan
        if (workspace/'failure.json').exists():
            archive=workspace/'previous-failures';archive.mkdir(exist_ok=True)
            _write(archive/f'{len(list(archive.glob("*.json")))+1:02d}.json',json.loads((workspace/'failure.json').read_text()))
    else:
        workspace.mkdir(exist_ok=False);output.mkdir()
    try:
        for task in 'ABC':
            if task not in results:results[task]=run_task(a.bundles/task,output/task,a.policy,a.cpu_budget,prices,resume=task in resume_tasks)
        export_models(output,workspace/'models')
        from ..solution_tracking import prepare_tools
        _write(workspace/'swanlab-metrics.json',prepare_tools(output))
        _write(workspace/'summary.json',dict(complete=True,policy=a.policy,explicit_continuation=a.resume,tasks=results,model_api_calls=sum(v['model_api_calls'] for v in results.values()),platform_evaluations=0))
        print('UNIFIED_RUN_COMPLETE',a.policy,flush=True)
    except Exception as error:
        message=str(error);key=os.environ.get('PYTRIO_API_KEY')
        if key:message=message.replace(key,'[REDACTED]')
        _write(workspace/'failure.json',dict(error_type=type(error).__name__,message=message));raise


if __name__=='__main__':main()
