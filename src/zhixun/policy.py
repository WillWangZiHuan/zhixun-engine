"""Bounded decisions from the competition's designated driver model."""
import json
import os
from pathlib import Path
import subprocess
import sys


MODEL = 'Qwen/Qwen3.8-27B'
ESTIMATORS = ('rf', 'extra_tree', 'lgbm')
PROMPT = '''You maintain a CPU industrial predictor under distribution drift.
Choose one experiment using ONLY the supplied observations and past experiment evidence.
You cannot see future labels, execute code, change safety gates, or increase budgets.
A distribution shift suggests a hypothesis; it does not establish a causal diagnosis.
Compare expected benefit, failed/rolled-back experiments, window size and training cost.
Gate evidence compares one tried candidate with its incumbent, not with untried actions.
Optional followup reports their paired loss on later observations: positive normalized_gain
means that tried candidate improved on its previous incumbent. This is different from gate gain.
Followup is delayed to exclude the fresh tuning/gate tail; absence is not evidence of success.
Historical gains come from different states and do not establish a universally best action.
Return exactly one JSON object, no markdown, with keys:
action: "train" or "wait";
window: an integer in eligible_windows for train, null for wait;
estimator: "rf", "extra_tree", or "lgbm" for train, null for wait;
hypothesis: a concise falsifiable explanation of the choice;
evidence_ids: integer IDs of supplied past experiments used, or [].
The evaluator alone decides promotion and rollback on separate evidence.'''


def validate_decision(value, request):
    expected = {'action', 'window', 'estimator', 'hypothesis', 'evidence_ids'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Driver output does not match the decision schema')
    if value['action'] not in ('train', 'wait'):
        raise ValueError('Unknown action')
    if not isinstance(value['hypothesis'], str) or not 1 <= len(value['hypothesis']) <= 1000:
        raise ValueError('A bounded experiment hypothesis is required')
    ids = value['evidence_ids']
    known = {e['id'] for e in request['experience']}
    if not isinstance(ids, list) or any(type(i) is not int or i not in known for i in ids):
        raise ValueError('Driver cited nonexistent or unavailable experience')
    if value['action'] == 'wait':
        if value['window'] is not None or value['estimator'] is not None:
            raise ValueError('Wait cannot contain a training action')
    elif (type(value['window']) is not int or value['window'] not in request['eligible_windows']
          or value['estimator'] not in ESTIMATORS):
        raise ValueError('Driver requested an unavailable window or estimator')
    return value


def decide(directory, request, *, timeout=120):
    """No local-rule replacement on API failure, invalid output or exhausted budget."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
    with (directory / 'worker.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'zhixun.policy_worker', str(directory.resolve())],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout,
                       env=dict(os.environ))
    response = json.loads((directory / 'response.json').read_text())
    decision = validate_decision(json.loads(response['text']), request)
    return decision, {k: response[k] for k in ('model', 'input_tokens', 'output_tokens', 'wall_seconds')}
