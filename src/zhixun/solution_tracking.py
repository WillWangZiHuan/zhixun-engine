"""Export and publish numeric records for the three current training recipes."""
from pathlib import Path
import argparse,json,math,os


def prepare(run):
    run=Path(run)
    summary=json.loads((run/'summary.json').read_text())
    if not summary.get('complete'):raise ValueError('Training run is incomplete')
    records=[]
    for task in 'ABC':
        folder=run/task
        events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
        if len(events)!=10 or [e['round'] for e in events]!=list(range(1,11)):
            raise ValueError('Expected ten contiguous rounds for each task')
        for event in events:
            n=event['round'];history=[json.loads(x) for x in (folder/f'round-{n:02d}/history.jsonl').read_text().splitlines()]
            for h in history:
                if task=='A' and h['kind']=='head_fit':
                    records.append({f'{task}/train/head_balanced_nll':h['balanced_cross_entropy_before'],f'{task}/train/head_reported_iteration':0})
                    records.append({f'{task}/train/head_balanced_nll':h['balanced_cross_entropy_after'],f'{task}/train/head_reported_iteration':h['optimizer_iterations']})
                else:
                    loss_name='residual_tree_mse' if task=='C' else 'core_loss'
                    iteration=h['tree_iteration'] if task=='C' else h['epoch']
                    records.append({f'{task}/train/{loss_name}':h['loss'],f'{task}/train/iteration':iteration,f'{task}/train/round':n})
            primary='macro_f1' if task=='A' else 'rmse'
            values={f'{task}/loop/round':n,f'{task}/loop/accepted':int(event['accepted']),
                    f'{task}/loop/candidate_{primary}':event['candidate_metrics'][primary],
                    f'{task}/loop/selected_{primary}':event['selected_metrics'][primary]}
            if task=='A':
                values[f'{task}/loop/selected_accuracy']=event['selected_metrics']['accuracy']
                for label,group in event['selected_metrics']['recall'].items():
                    if group['recall'] is not None:values[f'{task}/class/{label}/selected_recall']=group['recall']
            else:
                groups=event['candidate_groups'] if task=='B' else event['candidate_metrics']['groups']
                for label,group in groups.items():values[f'{task}/group/{label}/candidate_rmse']=group['rmse']
            records.append(values)
    for record in records:
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in record.values()):
            raise ValueError('Only finite numeric records may be published')
    return dict(format='zhixun.numeric-training-records.v1',
                config=dict(tasks=['A','B','C'],rounds_per_task=10,driver='fixed rules',public_validation_reused=True,
                            hidden_test_verified=False,model_api_calls=0,platform_evaluations=0,
                            head_loss_scope='initial and final balanced NLL; not all optimizer iterations'),records=records)


def prepare_agent(run):
    run=Path(run);summary=json.loads((run/'summary.json').read_text())
    if not summary['complete'] or summary['experiments']!=10:raise ValueError('Expected a completed ten-attempt code experiment')
    if 'actual_model_api_calls' in summary and summary['actual_model_api_calls']!=summary['model_calls']:
        raise ValueError('Offline decision replay must not be published as new model inference')
    events=json.loads((run/'events.json').read_text())
    if [e['trial'] for e in events]!=list(range(1,11)):raise ValueError('Incomplete experiment history')
    records=[]
    for e in events:
        if e['execution_status']=='completed':
            history=json.loads((run/f"trial-{e['trial']:02d}/result/training.json").read_text())['history']
            for h in history:records.append({'B/train/loss':h['loss'],'B/train/epoch':h['epoch'],'B/train/trial':e['trial']})
        values={'B/loop/trial':e['trial'],'B/loop/selected_rmse':e['selected_metrics']['rmse'],
                'B/loop/accepted':int(e['accepted']),'B/loop/execution_completed':int(e['execution_status']=='completed'),
                'B/loop/cpu_seconds':e['cpu_seconds']}
        if e['candidate_metrics'] is not None:values['B/loop/candidate_rmse']=e['candidate_metrics']['rmse']
        for group,metrics in e['selected_metrics']['groups'].items():values['B/group/'+group+'/selected_rmse']=metrics['rmse']
        records.append(values)
    records.append({'cost/model_calls':summary['model_calls'],'cost/input_tokens':summary['input_tokens'],
                    'cost/output_tokens':summary['output_tokens'],'cost/estimated_cny':summary['estimated_cny']})
    if any(type(v) not in (int,float) or not math.isfinite(v) for r in records for v in r.values()):raise ValueError('Finite numeric records required')
    return dict(format='zhixun.numeric-training-records.v1',config=dict(tasks=['B'],driver='training-code experiment',
                policy=summary['policy'],attempts=10,public_validation_reused=True,pretraining_overlap=True,
                hidden_test_verified=False,model_api_calls=summary['model_calls'],actual_bill_verified=False,
                platform_evaluations=0),records=records)


def prepare_tools(runs):
    """Numerical records from the selected-state tools, including failed attempts."""
    runs=Path(runs);records=[];total_calls=0
    for task in ('A','B','C'):
        root=runs/task;state=json.loads((root/'state.json').read_text())
        if state.get('format')!='zhixun.task-tools.v2' or not state.get('complete') or len(state['events'])!=10:raise ValueError('Completed v2 task runs required')
        total_calls+=state['model_api_calls']
        for event in state['events']:
            n=event['trial']
            if event['execution_status']=='completed':
                report=json.loads((root/f'trial-{n:02d}/result/training.json').read_text())
                for h in report['history']:
                    if h['kind']=='core_epoch':records.append({task+'/train/core_loss':h['loss'],task+'/train/epoch':h['epoch'],task+'/train/trial':n})
                    elif h['kind']=='residual_tree':records.append({task+'/train/residual_tree_mse':h['loss'],task+'/train/tree_iteration':h['tree_iteration'],task+'/train/trial':n})
                    elif h['kind']=='head_fit':
                        for phase,key in enumerate(('balanced_cross_entropy_before','balanced_cross_entropy_after')):records.append({task+'/train/head_balanced_nll':h[key],task+'/train/head_phase':phase,task+'/train/trial':n})
                    else:raise ValueError('Unknown training record kind')
            values={task+'/loop/trial':n,task+'/loop/accepted':int(event['accepted']),task+'/loop/execution_completed':int(event['execution_status']=='completed'),task+'/loop/child_cpu_seconds':event['child_cpu_seconds'],task+'/loop/local_cpu_seconds':event['local_cpu_seconds']}
            key='macro_f1' if task=='A' else 'rmse';values[task+'/loop/selected_'+key]=event['selected_metrics'][key]
            if event['candidate_metrics'] is not None:values[task+'/loop/candidate_'+key]=event['candidate_metrics'][key]
            if task=='A':
                for name,entry in event['selected_metrics']['recall'].items():
                    if entry['recall'] is not None:values[task+'/class/'+name+'/selected_recall']=entry['recall']
            else:
                for name,entry in event['selected_metrics']['groups'].items():values[task+'/group/'+name+'/selected_rmse']=entry['rmse']
            records.append(values)
    if any(type(v) not in (int,float) or not math.isfinite(v) for r in records for v in r.values()):raise ValueError('Finite numerical training records required')
    return dict(format='zhixun.numeric-training-records.v1',config=dict(tasks=['A','B','C'],driver='common experiment tools; decision provenance remains in proposals',attempts_per_task=10,public_validation_reused=True,hidden_test_verified=False,model_api_calls=total_calls,platform_evaluations=0),records=records)


def upload(payload, workspace, project, name, output):
    import swanlab
    payload=Path(payload);output=Path(output);data=json.loads(payload.read_text())
    if data['format']!='zhixun.numeric-training-records.v1':raise ValueError('Unexpected numeric record format')
    # Validate values before creating any cloud run. Text/source documents are not accepted.
    for record in data['records']:
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in record.values()):raise ValueError('Numeric records only')
    key=os.environ.get('SWANLAB_API_KEY')
    if not key:raise ValueError('SWANLAB_API_KEY must be supplied at runtime')
    api=swanlab.Api(api_key=key);cloud=api.project(workspace+'/'+project)
    if cloud.visibility!='PUBLIC':raise ValueError('Use the approved public experiment project')
    if name in {r.name for r in cloud.runs_get(all=True)}:raise ValueError('Run name already exists; inspect before retry')
    output.mkdir(parents=True,exist_ok=False)
    settings=swanlab.Settings(interactive=False,probe={k:False for k in ['hardware','runtime','requirements','conda','git','swanlab','monitor']},terminal={'proxy_type':'none'})
    run=swanlab.init(project=project,workspace=workspace,name=name,public=True,mode='online',on_online_error='raise',log_dir=str(output/'swanlog'),config=data['config'],settings=settings)
    receipt=dict(url=run.url,finished=False);(output/'receipt.json').write_text(json.dumps(receipt)+'\n');expected={}
    for step,record in enumerate(data['records'],1):
        swanlab.log(record,step=step)
        for k,v in record.items():expected.setdefault(k,[]).append(v)
    swanlab.finish();receipt['finished']=True;(output/'receipt.json').write_text(json.dumps(receipt)+'\n')
    cloud=api.project(workspace+'/'+project);uploaded=next(r for r in cloud.runs_get(all=True) if r.name==name)
    if cloud.visibility!='PUBLIC' or uploaded.state!='FINISHED':raise RuntimeError('Cloud record did not finish publicly')
    response=uploaded.metrics(list(expected),all=True);actual={r['key']:[x['value'] for x in r['metrics']] for r in response['list']}
    for k,values in expected.items():
        if len(actual[k])!=len(values) or not all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(actual[k],values)):raise ValueError('Cloud numeric readback mismatch: '+k)
    result=dict(complete=True,url=uploaded.url,state=uploaded.state,visibility=cloud.visibility,checked_series={k:len(v) for k,v in expected.items()},official_log_standard_verified=False,model_api_calls=0)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare-agent');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare-tools');p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('upload')
    for key in ('payload','workspace','project','name','output'):p.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if args.command in ('prepare','prepare-agent','prepare-tools'):
        data=prepare_tools(args.runs) if args.command=='prepare-tools' else prepare(args.run) if args.command=='prepare' else prepare_agent(args.run)
        with args.output.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
        print(json.dumps(dict(records=len(data['records']))))
    else:print(json.dumps(upload(args.payload,args.workspace,args.project,args.name,args.output)))


if __name__=='__main__':main()
