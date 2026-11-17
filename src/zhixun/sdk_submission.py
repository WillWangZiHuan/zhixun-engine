"""Workspace-v2 protocol: propose, continue, measure, select, persist, acknowledge."""
from pathlib import Path
from functools import lru_cache
import json
import os
import re
import sys
from contest_harness.errors import HarnessError
from baselines.data import json_value
from baselines.predict import predict as model_predict
from .checkpoint import load_checkpoint, core
from .experiment_agent.unified_protocol import parse
from .sdk_training import prepare, continue_candidate, write, digest
from .round_prompt import PROMPT

PREDICT_ARGV = ['python3', '/app/harness.py', 'predict']
SELECT_PROMPT = '''Choose between the supplied parent artifact and the new candidate using the
actual platform candidate receipt, task contract, experiment and history. The host owns
measurement; local diagnostics are not platform scores. Do not invent missing measurements,
causality, hidden-test gains or call IDs. Return exactly one JSON object containing
selected_artifact_id (one of allowed_artifact_ids) and reason (nonempty, <=4000 UTF-8 bytes).
Explain your decision from the available evidence. No Markdown or other fields.'''


def safe_error(error):
    text = '%s: %s' % (type(error).__name__, error)
    text = re.sub(r'(?i)(bearer\s+)[^\s]+', r'\1[REDACTED]', text)
    text = re.sub(r'(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}', '[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;&]+', r'\1[REDACTED]', text)
    return text[:2000]


def call_model(rnd, root, name, prompt, view):
    folder = root/name
    folder.mkdir()
    messages = [{'role':'system', 'content':prompt},
                {'role':'user', 'content':json.dumps(view, ensure_ascii=False, allow_nan=False)}]
    write(folder/'messages.json', messages)
    # SDK supplies the authorized output limit and the host prepends its prompt once.
    receipt = rnd.model(messages, temperature=0)
    write(folder/'receipt.json', dict(call_id=receipt.call_id, response=receipt.response))
    response = receipt.response
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('Expected one model choice')
    if choices[0].get('finish_reason') == 'length':
        raise ValueError('Model reply truncated by output limit')
    text = receipt.text
    if not text.strip() or len(text.encode('utf-8')) > 131072:
        raise ValueError('Expected bounded nonempty decision text')
    return receipt, text


def selection(text, allowed):
    value = json_value(text)
    if not isinstance(value, dict) or set(value) != {'selected_artifact_id', 'reason'}:
        raise ValueError('Selection must contain selected_artifact_id and reason')
    if value['selected_artifact_id'] not in allowed:
        raise ValueError('Model selected an artifact outside this decision')
    reason = value['reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode('utf-8')) > 4000:
        raise ValueError('Selection reason must be nonempty and <=4000 bytes')
    return value


def train(rnd):
    stage = 'input'
    root = None
    try:
        # /workspace exists in the host; every round creates its own new child directory.
        work = Path(os.environ.get('CONTEST_WORK', '/workspace'))
        root = work/('zhixun-%s-%02d' % (rnd.dataset_key, rnd.round_index))
        root.mkdir()
        checkpoints = rnd.parent.descriptor['checkpoint_paths']
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise ValueError('This model requires one declared complete checkpoint')
        parent = rnd.parent.path(checkpoints[0])
        task = rnd.task.get('task')
        prepared = prepare(parent, rnd.data.train_path, rnd.data.validation_path, rnd.dataset_key, task)
        model, _, _, _, hashes, view = prepared
        stage = 'history'
        history = rnd.history()
        # Preserve SDK history verbatim: it is not the old promote/keep/retain schema.
        write(root/'platform-history.json', history)
        stage = 'state_read'
        state = rnd.state_workspace(root/'state')
        previous = json_value((state/'last-round.json').read_text()) if (state/'last-round.json').exists() else None
        view.update(round_index=rnd.round_index, actual_parent_file_sha256=hashes['parent'],
                    parent_checkpoint_receipt=rnd.parent.checkpoint_sha256,
                    parent_artifact_id=rnd.parent.artifact_id, task_contract=rnd.task,
                    actual_platform_history=history, previous_native_state=previous,
                    required_next='experiment')
        stage = 'proposal_model'
        proposal_receipt, text = call_model(rnd, root, 'proposal', PROMPT, view)
        stage = 'proposal_validation'
        action = parse(text)
        if action['tool'] != 'experiment':
            raise ValueError('Training requires one experiment proposal')
        proposal = action['proposal']
        if len(proposal['hypothesis'].encode('utf-8')) > 4000:
            raise ValueError('Hypothesis exceeds SDK limit')
        write(root/'proposal/action.json', action)
        stage = 'continuation'
        candidate, report = continue_candidate(root, prepared, proposal)
        change = ('Continued supplied parent from %d to %d Adam steps; full model kind=%s; '
                  'program sha256=%s; no device metadata used.' %
                  (core(model).meta['steps'], report['core_steps_after'], report['model_kind'], report['program_sha256']))
        configuration = dict(program_sha256=report['program_sha256'],
                             plan=report['numerical']['plan'], supplement=proposal.get('supplement'),
                             readout_expression=proposal.get('readout_expression'),
                             parent_file_sha256=hashes['parent'], proposal_call_id=proposal_receipt.call_id)
        seed = model.meta['seed']
        # Keep exact proposal and call receipt alongside the model, separate from protocol stdout.
        write(candidate/'proposal.json', action)
        write(candidate/'model-receipt.json', dict(call_id=proposal_receipt.call_id,
                                                  response=proposal_receipt.response))
        stage = 'candidate_submit'
        submitted = rnd.submit_candidate(candidate, hypothesis=proposal['hypothesis'],
                    change_summary=change, configuration=configuration, seed=seed,
                    checkpoint_paths=['checkpoint.npz'], code_paths=[], predict_argv=PREDICT_ARGV)
        write(root/'candidate-receipt.json', submitted)
        stage = 'selection_model'
        allowed = [rnd.parent.artifact_id, submitted['artifact_id']]
        select_view = dict(allowed_artifact_ids=allowed, parent_artifact=rnd.parent.receipt,
                           candidate_receipt=submitted, task_contract=rnd.task,
                           actual_platform_history=history, proposal=action,
                           local_diagnostics={'scope':'not platform measurements',
                               'before':report['local_validation_before'],
                               'after':report['local_validation_after']})
        choice_receipt, text = call_model(rnd, root, 'selection', SELECT_PROMPT, select_view)
        stage = 'selection_validation'
        chosen = selection(text, allowed)
        write(root/'selection/action.json', chosen)
        stage = 'state_save'
        write(state/'last-round.json', dict(dataset_key=rnd.dataset_key, round_index=rnd.round_index,
              supplied_parent_artifact_id=rnd.parent.artifact_id, candidate_receipt=submitted,
              selected_artifact_id=chosen['selected_artifact_id'], reason=chosen['reason'],
              proposal=action, selection_receipt={'call_id':choice_receipt.call_id, 'response':choice_receipt.response},
              model_call_ids=list(rnd.call_ids), history_ids=[item['id'] for item in history]))
        rnd.save_state(state)
        stage = 'finish'
        rnd.finish(selected_artifact_id=chosen['selected_artifact_id'], hypothesis=proposal['hypothesis'],
                   change_summary=change, reason=chosen['reason'], configuration=configuration, seed=seed,
                   history_ids=[item['id'] for item in history], model_call_ids=list(rnd.call_ids))
        # No protocol frames after round_ack.
    except Exception as error:
        detail = dict(stage=stage, error=safe_error(error), model_call_ids=list(rnd.call_ids),
                      automatic_retry=False, fixed_rule_fallback=False)
        print(json.dumps(detail, ensure_ascii=False), file=sys.stderr, flush=True)
        if root is not None and root.is_dir():
            write(root/'failure.json', detail)
        raise HarnessError('ZHIXUN_'+stage.upper(), detail['error']) from None


@lru_cache(maxsize=1)
def prediction_model(path):
    return load_checkpoint(Path(path))


def predict(sample, model_dir):
    # Host declares an exact selected checkpoint; never search /app or choose our latest candidate.
    raw = os.environ.get('CONTEST_ARTIFACT_PATH')
    if not raw:
        raise HarnessError('ARTIFACT_PATH_REQUIRED')
    path, directory = Path(raw), Path(model_dir)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise HarnessError('ARTIFACT_OUTSIDE_MODEL_MOUNT')
    current = path
    while current != directory and current != current.parent:
        if current.is_symlink():
            raise HarnessError('ARTIFACT_SYMLINK')
        current = current.parent
    model = prediction_model(str(path.resolve()))
    if model.meta['task'] != sample.task or (sample.dataset_key is not None and model.meta['profile'] != sample.dataset_key):
        raise HarnessError('PREDICTION_TASK_MISMATCH')
    return model_predict(model, sample.inputs)
