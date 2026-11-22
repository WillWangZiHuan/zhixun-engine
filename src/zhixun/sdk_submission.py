"""Workspace-v2 protocol: propose, continue, measure, select, persist, acknowledge."""
from pathlib import Path
from functools import lru_cache
import json
import math
import os
import re
import sys
from contest_harness.errors import HarnessError
from baselines.data import json_value
from baselines.predict import predict as model_predict
from .checkpoint import MAX_BYTES, load_checkpoint, core
from .experiment_agent.unified_protocol import parse
from .sdk_training import RejectedProposal, prepare, continue_candidate, write
from .round_prompt import A_PROMPT, PROMPT

PREDICT_ARGV = ['python3', '/app/harness.py', 'predict']
MAX_OUTPUT_TOKENS = 16_384
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
    max_tokens = min(rnd.max_output_tokens, MAX_OUTPUT_TOKENS)
    print(json.dumps(dict(dataset=rnd.dataset_key, round_index=rnd.round_index, stage=name,
                          host_max_output_tokens=rnd.max_output_tokens, max_tokens=max_tokens,
                          enable_thinking=False)),
          file=sys.stderr, flush=True)
    receipt = rnd.model(messages, temperature=0, max_tokens=max_tokens, enable_thinking=False)
    write(folder/'receipt.json', dict(call_id=receipt.call_id, response=receipt.response))
    return receipt


def decision(rnd, root, name, prompt, view, evaluate):
    """A confirmed unusable reply can be revised before any host-side commit."""
    feedback = None
    for attempt in range(1, 4):
        label = name if attempt == 1 else f'{name}-{attempt}'
        request = view if feedback is None else {**view, 'revision_request': feedback}
        receipt = call_model(rnd, root, label, prompt, request)
        folder = root/label
        text = receipt.text
        try:
            choices = receipt.response.get('choices')
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise RejectedProposal('Return one decision in the assistant message.')
            if choices[0].get('finish_reason') == 'length':
                raise RejectedProposal('Output was truncated. Return a shorter complete decision.')
            if not text.strip() or len(text.encode('utf-8')) > 131072:
                raise RejectedProposal('Return a bounded nonempty decision.')
            return receipt, evaluate(folder, text)
        except RejectedProposal as error:
            feedback = dict(rejected_call_id=receipt.call_id, error=safe_error(error),
                            previous_reply=text[:16000],
                            instruction='Correct the reported error and return one complete new decision.')
            write(folder/'rejection.json', feedback)
            if attempt == 3:
                raise


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
        prompt = A_PROMPT if rnd.dataset_key == 'A' else PROMPT
        def experiment(folder, text):
            try:
                action = parse(text)
                if action['tool'] != 'experiment':
                    raise ValueError('Training requires one experiment proposal')
                hypothesis = action['proposal'].get('hypothesis')
                if not isinstance(hypothesis, str) or not 0 < len(hypothesis.encode('utf-8')) <= 4000:
                    raise ValueError('Expected a nonempty hypothesis of at most 4000 UTF-8 bytes')
            except (TypeError, ValueError, RecursionError) as exc:
                raise RejectedProposal(str(exc)) from exc
            write(folder/'action.json', action)
            candidate, report = continue_candidate(folder, prepared, action['proposal'])
            return action, candidate, report

        proposal_receipt, (action, candidate, report) = decision(
            rnd, root, 'proposal', prompt, view, experiment)
        proposal = action['proposal']
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
        if sum(path.stat().st_size for path in candidate.rglob('*') if path.is_file()) > MAX_BYTES:
            raise ValueError('Complete candidate exceeds 64 MiB')
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
        def choose(folder, text):
            try:
                chosen = selection(text, allowed)
            except (TypeError, ValueError, RecursionError) as exc:
                raise RejectedProposal(str(exc)) from exc
            write(folder/'action.json', chosen)
            return chosen

        choice_receipt, chosen = decision(rnd, root, 'selection', SELECT_PROMPT, select_view, choose)
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
                      rpc_replayed=False)
        print(json.dumps(detail, ensure_ascii=False), file=sys.stderr, flush=True)
        if root is not None and root.is_dir():
            write(root/'failure.json', detail)
        if isinstance(error, HarnessError):
            raise
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
    result = model_predict(model, sample.inputs)
    if sample.task == 'classification':
        scores = result['outputs']
        best = max(row['score'] for row in scores)
        tied = [row for row in scores if row['score'] == best]
        # Match the trained classifier's argmax ordering on exact ties.
        if len(tied) > 1 and best > 0:
            for row in tied[1:]:
                row['score'] = math.nextafter(best, 0.0)
    return result
