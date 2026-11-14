"""One bounded application attempt; provider usage survives invalid actions."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from ..joint_driver import MODEL, uncached_estimate
from .protocol import parse
from .phase_runtime import AttemptBudget, run_attempt

FORMAT = 'zhixun.unified-phase.v1'
MAX_INPUT_TOKENS = 12000
MAX_OUTPUT_TOKENS = 2048
INITIALIZATION_SECONDS = 60
SAMPLING_SECONDS = 60

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def validate_prices(prices):
    uncached_estimate(0, 0, prices)
    if not isinstance(prices.get('checked_at'), str) or not prices['checked_at'].strip():
        raise ValueError('Dated designated-model price required before launch')

def validate_action(response, request, parser):
    if response.get('action_error'):
        raise ValueError('Worker could not extract a complete action')
    incoming, outgoing = response['input_tokens'], response['output_tokens']
    if type(response.get('local_input_tokens')) is not int or incoming != response['local_input_tokens'] or incoming > MAX_INPUT_TOKENS or outgoing > MAX_OUTPUT_TOKENS:
        raise ValueError('Token contract')
    ids = response.get('output_token_ids')
    if not isinstance(ids, list) or any(type(n) is not int or n < 0 for n in ids) or len(ids) != outgoing:
        raise ValueError('Output token sequence contract')
    if response.get('stop_reason') != 'stop' or not isinstance(response.get('text'), str):
        raise ValueError('Incomplete generation')
    value = parser(response['text'])
    allowed = {'experiment': {'experiment'}, 'finish': {'finish'},
               'one read/analyze or experiment': {'read', 'analyze', 'experiment'}}
    if request['required_next'] not in allowed or value['tool'] not in allowed[request['required_next']]:
        raise ValueError('Tool does not match experiment phase')
    return value

def decide(directory, request, *, python, prices, task_deadline,
           worker_module='zhixun.experiment_worker', decision_parser=parse):
    validate_prices(prices)
    if task_deadline <= time.monotonic():
        raise TimeoutError('Original task deadline expired before attempt')
    directory = Path(directory).resolve()
    calls = sorted(directory.parent.glob('call-*'))
    if directory.exists():
        raise FileExistsError('Existing call is immutable; no replay or second generation')
    if len(calls) >= 21 or directory.name != f'call-{len(calls)+1:02d}':
        raise ValueError('Sequential 21-directory call budget')
    if [p.name for p in calls] != [f'call-{i+1:02d}' for i in range(len(calls))]:
        raise ValueError('Incomplete call history')
    prompt = importlib.import_module(worker_module).PROMPT
    directory.mkdir(parents=True, exist_ok=False)
    write(directory/'request.json', request)
    write(directory/'price.json', prices)
    (directory/'prompt.txt').write_text(prompt)
    write(directory/'manifest.json', dict(format=FORMAT, worker_module=worker_module,
        files={n: hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ['request.json','price.json','prompt.txt']}))
    env = dict(os.environ)
    env.pop('SWANLAB_API_KEY', None)
    source = Path(__file__).resolve().parents[2]
    env['PYTHONPATH'] = str(source) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONNOUSERSITE'] = '1'
    transport = None
    failure = None
    value = None
    billing = dict(format=FORMAT,status='no_sampling_request',actual_bill=False,input_tokens=0,output_tokens=0,estimated_uncached_cny=0)
    try:
        transport = run_attempt([str(python),'-B','-m',worker_module,str(directory)],directory/'transport',
            AttemptBudget(21, used=len(calls)), task_deadline=task_deadline,
            initialization_seconds=INITIALIZATION_SECONDS,sampling_seconds=SAMPLING_SECONDS,env=env)
        if transport['sampling_authorizations']:
            billing = dict(format=FORMAT,status='unknown',actual_bill=False,input_tokens=None,output_tokens=None,
                estimated_uncached_cny=None,reserved_upper_cny=uncached_estimate(MAX_INPUT_TOKENS,MAX_OUTPUT_TOKENS,prices))
        response_file = directory/'transport/response.json'
        usage_file = directory/'provider-usage.json'
        # The worker writes usage before indexing/decoding, including a decode timeout.
        if response_file.exists() or usage_file.exists():
            response = json.loads((response_file if response_file.exists() else usage_file).read_text())
            write(directory/'response.json', response)
            if response['model'] != MODEL:
                raise ValueError('Provider model does not match quoted model')
            cost = uncached_estimate(response['input_tokens'],response['output_tokens'],prices)
            billing = dict(format=FORMAT,status='usage_returned',model=MODEL,
                input_tokens=response['input_tokens'],output_tokens=response['output_tokens'],
                estimated_uncached_cny=cost,actual_bill=False,price_checked_at=prices['checked_at'],
                wall_seconds=transport['wall_seconds'],local_cpu_seconds=response.get('local_cpu_seconds'))
            if transport['status'] != 'response_returned':
                raise ValueError('Late or invalid transport result cannot execute an action')
            value = validate_action(response, request, decision_parser)
            if time.monotonic() >= task_deadline:
                value = None
                raise TimeoutError('Action parsing crossed the original task deadline')
        else:
            failure = 'Transport_'+transport['status']
    except Exception as exc:
        failure = type(exc).__name__
    billing.update(sampling_authorizations=0 if transport is None else transport['sampling_authorizations'],
        local_sampling_started_markers=0 if transport is None else transport['sampling_started_markers'],
        http_dispatch_count=None,application_attempt=directory.name)
    write(directory/'billing.json',billing)
    write(directory/'result.json',dict(format=FORMAT,status='decision_returned' if value is not None else 'failed',
        failure=failure,transport=transport,billing=billing))
    print(json.dumps({'api_consumption':billing,'profile':request['profile'],'round':request['next_round']}),flush=True)
    if value is None:
        write(directory/'error.json',dict(format=FORMAT,error_type=failure,transport_status=None if transport is None else transport['status']))
        raise RuntimeError('Designated-model attempt failed; see preserved receipts')
    write(directory/'decision.json',value)
    return value,billing
