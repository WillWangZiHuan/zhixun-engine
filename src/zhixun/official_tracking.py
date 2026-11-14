"""Prepare a path-free SwanLab payload from immutable official continuation records."""
import hashlib
import json
from pathlib import Path


def prepare(source):
    source=Path(source)
    protocol=json.loads((source/'protocol.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    events=[json.loads(s) for s in (source/'events.jsonl').read_text().splitlines()]
    decisions_path=source/'decisions.jsonl'
    decisions=[json.loads(s) for s in decisions_path.read_text().splitlines()] if decisions_path.exists() else []
    if summary['status']!='completed' or len(events)!=summary['rounds']:
        raise ValueError('only complete, consistent runs can be recorded')
    if decisions and len(decisions)!=len(events):
        raise ValueError('decision history is incomplete')
    primary='macro_f1' if protocol['profile']=='A' else 'rmse'
    config={k:protocol[k] for k in ('protocol','profile','task','selection','rounds','epochs_per_candidate',
            'hyperparameters','same_public_data_every_round','selection_data_is_evaluation_data',
            'independent_generalization_test','official_platform_protocol','source_hashes','runner_sha256')}
    config.update(recording='retrospective export; original data and timestamps retained locally',
                  official_log_standard_verified=False,model_api_calls=summary['model_api_calls'],platform_evaluations=0,
                  source_events_sha256=hashlib.sha256((source/'events.jsonl').read_bytes()).hexdigest())
    if protocol['selection'] in ('pytrio', 'joint_model'):
        config['driver_sources']=protocol['driver_sources']
        config['model']=protocol['price_record']['model']
        config['price_record']={k:protocol['price_record'][k] for k in ('checked_at','model','currency','prefill_per_million','sample_per_million')}
    if 'training_runtime_hashes' in protocol:
        config['training_runtime_hashes']=protocol['training_runtime_hashes']
    records=[{'metrics':{f'initial/{k}':v for k,v in summary['initial_metrics'].items()},
              'kind':'initial','evidence':{'initial_metrics':summary['initial_metrics']}}]
    for n,event in enumerate(events,1):
        if event['round']!=n:raise ValueError('round sequence is not continuous')
        if decisions:
            d=decisions[n-1]
            if d['round']!=n or d['parent_sha256']!=event['parent_sha256']:
                raise ValueError('proposal does not bind the executed parent')
            records.append({'kind':'proposal','metrics':{'proposal/round':n,'proposal/learning_rate':d['learning_rate']},
                            'evidence':{**d,'history_field_semantics':('based_on_rounds lists available prior rounds; evidence_rounds are model-cited references, not proof of reasoning' if protocol['selection'] in ('pytrio', 'joint_model') else 'based_on_rounds lists available prior rounds; rule consumes recorded history')}})
        if protocol['selection']=='joint_model' or (protocol['selection']=='pytrio' and n>1):
            api=source/f'api-{n:02d}'
            bill=json.loads((api/'billing.json').read_text())
            if bill['status']!='usage_returned':raise ValueError('unresolved model billing cannot be reported as complete')
            records.append({'kind':'api','metrics':{'api/round':n,'api/input_tokens':bill['input_tokens'],
                'api/output_tokens':bill['output_tokens'],'api/estimated_uncached_cny':bill['estimated_uncached_cny'],
                'api/wall_seconds':bill['wall_seconds']},'evidence':{'resolved_action':json.loads((api/'decision.json').read_text()), 'usage_status':bill['status'], 'actual_bill':False, 'raw_request_and_response':'Retained locally; not uploaded.'}})
        folder=source/f'round-{n:02d}'
        report=json.loads((folder/'training.json').read_text())
        if report['parent_checkpoint_sha256']!=event['parent_sha256'] or report['checkpoint_sha256']!=event['candidate_sha256']:
            raise ValueError('training report and selection evidence disagree')
        history=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
        if len(history)!=report['configuration']['requested_epochs']:
            raise ValueError('incomplete epoch history')
        for h in history:
            metrics={'train/round':n,'train/epoch':h['epoch'],'train/optimizer_step':h['step'],
                     'train/loss':h['loss'],'train/learning_rate':report['learning_rate']}
            metrics.update({f'train/validation_{k}':v for k,v in h['validation'].items()})
            records.append({'kind':'train','metrics':metrics,'evidence':{
                'history':h,'parent_sha256':report['parent_checkpoint_sha256'],
                'checkpoint_sha256':report['checkpoint_sha256'],'original_created_at':report['created_at'],
                'train_sha256':report['train_sha256'],'validation_sha256':report['validation_sha256'],
                'numpy_version':report['numpy_version'],'python_version':report['python_version'],
                **{k:report[k] for k in ('loss_mode','group_counts_train','weight_formula') if k in report}}})
        metrics={'loop/round':n,'loop/accepted':int(event['accepted']),
                 'loop/candidate_primary':event['candidate_metrics'][primary],
                 'loop/selected_primary':event['selected_metrics'][primary],
                 'loop/selected_optimizer_steps':event['selected_steps'],
                 'loop/elapsed_seconds':event['elapsed_seconds']}
        records.append({'kind':'selection','metrics':metrics,'evidence':event})
    if sum(rec['kind']=='api' for rec in records) != summary['model_api_calls']:
        raise ValueError('API evidence count differs from completed run summary')
    return {'name':f"official-public-{protocol['profile']}-{protocol['selection']}-20261111",
            'config':config,'records':records,
            'counts':{kind:sum(r['kind']==kind for r in records) for kind in (('initial','proposal','train','selection','api') if protocol['selection'] in ('pytrio', 'joint_model') else ('initial','proposal','train','selection'))}}
