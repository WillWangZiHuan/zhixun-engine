"""Bounded loss/LR proposals from this run's causal group-diagnostic prefix."""
import json, math
MODEL = 'Qwen/Qwen3.8-27B'
MAX_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 256
PROMPT = """Choose exactly one listed action_id for a CPU regression continuation experiment.
Each candidate specifies loss_mode and learning_rate. Copy its action_id verbatim; never output numeric parameters or invent candidates.
none uses ordinary MSE; balanced uses N/(G*n_g) train-group weights, sample mean one.
One epoch starts from selected_checkpoint; architecture, Adam state, scaler, seed and batch size stay fixed.
The fixed gate accepts only lower overall validation RMSE. Group errors diagnose tradeoffs; they do not change the gate.
Use only this run's completed observations. Initial observation is 0. No future or other-run results are supplied.
History columns describe candidates and parent/selected aliases. RMSE/MAE use physical units; training loss uses normalized targets and differs across objectives.
Do not infer causal causes from aggregate metrics. Public validation is reused, so do not claim unseen generalization.
Return one compact JSON object with exactly action_id, hypothesis, evidence_rounds.
hypothesis: one testable sentence at most 120 characters, no numerical recap.
evidence_rounds: 1-3 distinct completed integer IDs; 0 denotes initial. No code, markdown or other keys."""


def _base_request(profile, initial_learning_rate, selected_checkpoint, history, *, seed, initial_metrics, initial_groups, training_group_counts, initial_checkpoint):
    if profile not in ('B','C') or len(history)>9 or [e['round'] for e in history]!=list(range(1,len(history)+1)):
        raise ValueError('contiguous zero-to-nine completed rounds required')
    if type(initial_learning_rate) not in (float,int) or not math.isfinite(initial_learning_rate) or not 0<initial_learning_rate<=.5:
        raise ValueError('invalid initial rate')
    expected_parent=history[-1]['selected_sha256'] if history else initial_checkpoint
    if selected_checkpoint!=expected_parent:raise ValueError('selected parent mismatch')
    keys=sorted(initial_groups)
    if not keys or not training_group_counts or any(type(v) is not int or v<=0 for v in training_group_counts.values()):raise ValueError('invalid groups')
    tried={(e['loss_mode'],e['learning_rate']) for e in history if e['parent_sha256']==selected_checkpoint}
    actions=[[mode,initial_learning_rate*2.**k] for k in [0,1]+list(range(-1,-10,-1)) for mode in ('none','balanced') if (mode,initial_learning_rate*2.**k) not in tried]
    aliases={initial_checkpoint:0}
    rows=[]
    for e in history:
        if set(e['candidate_groups'])!=set(keys) or e['parent_sha256'] not in aliases:raise ValueError('invalid historical groups or parent')
        aliases[e['candidate_sha256']]=e['round']
        if e['selected_sha256'] not in aliases:raise ValueError('unobserved selected checkpoint')
        rows.append([e['round'],aliases[e['parent_sha256']],e['loss_mode'],e['learning_rate'],round(e['training_loss'],6),round(e['candidate_metrics']['rmse'],6),round(e['candidate_metrics']['mae'],6),[round(e['candidate_groups'][g]['rmse'],6) for g in keys],e['accepted'],aliases[e['selected_sha256']]])
    request={'profile':profile,'seed':seed,'next_round':len(history)+1,'remaining_training_epochs':10-len(history),'target_unit':'minutes' if profile=='B' else 'cycles','selected_checkpoint':selected_checkpoint,'selected_alias':aliases[selected_checkpoint],
        'allowed_actions':actions,'training_group_counts':training_group_counts,'validation_group_order':keys,
        'initial':{'observation':0,'rmse':initial_metrics['rmse'],'mae':initial_metrics['mae'],'groups':{g:initial_groups[g] for g in keys}},
        'history_columns':['round','parent_alias','loss_mode','learning_rate','training_loss','rmse','mae','group_rmse','accepted','selected_alias'],'history':rows}
    json.dumps(request,allow_nan=False)
    return request


def indexed_request(request):
    # IDs bind explicit original pairs, preserving candidate order and available actions.
    request = json.loads(json.dumps(request,allow_nan=False))
    request['candidates'] = [{'action_id':f'a{i:02d}','loss_mode':mode,'learning_rate':rate} for i,(mode,rate) in enumerate(request.pop('allowed_actions'))]
    return request


def build_request(*args, **kwargs):
    return indexed_request(_base_request(*args, **kwargs))


def parse_decision(text, request):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    value=json.loads(text,object_pairs_hook=unique)
    if not isinstance(value,dict) or set(value)!={'action_id','hypothesis','evidence_rounds'}:raise ValueError('invalid indexed schema')
    if not isinstance(value['action_id'],str):raise ValueError('action ID must be an exact string')
    matches=[a for a in request['candidates'] if a['action_id']==value['action_id']]
    if len(matches)!=1:raise ValueError('action ID is not an available candidate')
    if not isinstance(value['hypothesis'],str) or not 1<=len(value['hypothesis'].strip())<=120:raise ValueError('invalid hypothesis')
    ids=value['evidence_rounds']
    if not isinstance(ids,list) or not 1<=len(ids)<=3 or any(type(i) is not int or not 0<=i<request['next_round'] for i in ids) or len(set(ids))!=len(ids):raise ValueError('invalid completed evidence')
    return {**matches[0],'hypothesis':value['hypothesis'],'evidence_rounds':ids}


def uncached_estimate(input_tokens, output_tokens, prices):
    if any(type(n) is not int or n < 0 for n in (input_tokens,output_tokens)):
        raise ValueError('invalid token usage')
    if prices['model']!=MODEL or prices['currency']!='CNY':
        raise ValueError('price record does not match the model/currency')
    for k in ('prefill_per_million','sample_per_million'):
        if type(prices[k]) not in (float,int) or not math.isfinite(prices[k]) or prices[k]<0:
            raise ValueError('invalid price record')
    return (input_tokens*prices['prefill_per_million']+output_tokens*prices['sample_per_million'])/1e6


def decide(directory, request, *, python, prices, timeout=120):
    """One isolated sampling attempt. Invalid output or unknown billing stops the run."""
    import os
    from pathlib import Path
    import subprocess
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n')
    (directory/'price.json').write_text(json.dumps(prices, ensure_ascii=False, indent=2)+'\n')
    env = dict(os.environ)
    env.pop('SWANLAB_API_KEY', None)
    failure = None
    try:
        with (directory/'worker.log').open('w') as log:
            subprocess.run([str(python), '-B', '-m', 'zhixun.joint_worker', str(directory)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout, env=env)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        failure = type(exc).__name__
    response_path = directory/'response.json'
    response = None
    billing = None
    if response_path.exists():
        try:
            response = json.loads(response_path.read_text())
            if response['model'] != MODEL:
                raise ValueError('response model has no matching quote')
            cost = uncached_estimate(response['input_tokens'], response['output_tokens'], prices)
            billing = {'status':'usage_returned', 'model':response['model'],
                       'input_tokens':response['input_tokens'],'output_tokens':response['output_tokens'],
                       'estimated_uncached_cny':cost,'actual_bill':False,
                       'price_checked_at':prices['checked_at'], 'wall_seconds':response['wall_seconds']}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            # A truncated/malformed response cannot establish a bill. Stop without
            # discarding the file or guessing tokens/prices from partial fields.
            failure = failure or type(exc).__name__
            billing = {'status':'unknown', 'actual_bill':False, 'error_type':type(exc).__name__}
    else:
        stages = []
        progress_invalid = False
        try:
            progress = directory/'progress.jsonl'
            if progress.exists():
                for line in progress.read_text().splitlines():
                    stages.append(json.loads(line)['stage'])
        except (ValueError, KeyError, TypeError, OSError):
            progress_invalid = True
        billing = {'status':'unknown' if progress_invalid or 'sampling_requested' in stages or 'sampling_returned' in stages or 'response_saved' in stages else 'no_sampling_request',
                   'actual_bill':False, 'error_type':failure or 'MissingResponse'}
    (directory/'billing.json').write_text(json.dumps(billing,indent=2)+'\n')
    # Emitted before decision validation, so even rejected responses retain their consumption.
    print(json.dumps({'api_consumption':billing,'profile':request['profile'],'round':request['next_round']}),flush=True)
    if failure or not response_path.exists():
        raise RuntimeError('Designated-model attempt failed; see preserved billing and redacted logs')
    if response['model'] != MODEL or response['input_tokens'] != response['local_input_tokens'] or response['input_tokens'] > MAX_INPUT_TOKENS or response['output_tokens'] > MAX_OUTPUT_TOKENS:
        raise ValueError('Provider response violates model or token contract')
    if response['stop_reason'] != 'stop':
        raise ValueError('Model output did not finish normally; preserved usage includes this failed attempt')
    value = parse_decision(response['text'], request)
    (directory/'decision.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value, billing
