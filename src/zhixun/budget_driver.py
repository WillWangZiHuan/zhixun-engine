"""Bounded budget decisions with preserved per-call consumption records."""
import json
from .joint_driver import MODEL,MAX_INPUT_TOKENS,uncached_estimate
from .budget_protocol import PROMPT,parse_decision
MAX_OUTPUT_TOKENS=32

def decide(directory, request, *, python, prices, timeout=120):
    """One isolated sampling attempt. Invalid output or unknown billing stops the run."""
    import os
    from pathlib import Path
    import subprocess
    project=Path(__file__).resolve().parents[2]
    attempts=0
    for version in ('v1','v2'):
        for f in (project/'runs'/f'official-budget-devices-{version}').glob('*-llm/api-*/progress.jsonl'):
            if any(json.loads(line).get('stage')=='sampling_requested' for line in f.read_text().splitlines()):attempts+=1
    if attempts>=30:raise RuntimeError('Total 30-call study limit reached; no new request sent')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n')
    (directory/'price.json').write_text(json.dumps(prices, ensure_ascii=False, indent=2)+'\n')
    env = dict(os.environ)
    env.pop('SWANLAB_API_KEY', None)
    failure = None
    try:
        with (directory/'worker.log').open('w') as log:
            subprocess.run([str(python), '-B', '-m', 'zhixun.budget_worker', str(directory)],
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
