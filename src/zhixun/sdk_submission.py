"""Workspace-v2 protocol: propose, continue, measure, select, persist, acknowledge."""
from pathlib import Path
from functools import lru_cache
import json
import math
import os
import re
import sys
import time
from contest_harness.errors import HarnessError
from baselines.data import json_value
from baselines.predict import predict as model_predict
from baselines.model import Model

MAX_BYTES = 64 * 1024 * 1024
from .experiment_agent.unified_protocol import parse
from .sdk_training import RejectedProposal, prepare, continue_candidate, write
from .round_prompt import TASK_PROPOSAL_PROMPTS, TASK_SELECTION_HINTS

PREDICT_ARGV = ['python3', '/app/harness.py', 'predict']
MAX_OUTPUT_TOKENS = 16_384
SELECT_PROMPT = '''Choose between the supplied parent artifact and the new candidate using the
actual platform candidate receipt, task contract, experiment and history. The host owns
measurement; local diagnostics are not platform scores. Do not invent missing measurements,
causality, hidden-test gains or call IDs. Return exactly one JSON object containing
selected_role (exactly "parent" or "candidate") and reason (nonempty, <=4000 UTF-8 bytes). The harness binds your chosen role to this round’s real host receipt.
Explain your decision from the available evidence. No Markdown or other fields.'''


def safe_error(error):
    text = '%s: %s' % (type(error).__name__, error)
    text = re.sub(r'(?i)(bearer\s+)[^\s]+', r'\1[REDACTED]', text)
    text = re.sub(r'(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}', '[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;&]+', r'\1[REDACTED]', text)
    return text[:2000]


def call_model(rnd, root, name, prompt, view, feedback):
    folder = root/name
    folder.mkdir()
    messages = [{'role':'system', 'content':prompt},
                {'role':'user', 'content':json.dumps(view, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}]
    if feedback is not None:
        diagnostic = {k: v for k, v in feedback.items() if k != 'previous_reply'}
        action = 'Revise the failed decision using the execution error. Return one complete corrected JSON decision.'
        if 'Regression sample_weights receives STANDARDIZED y' in diagnostic['error']:
            action = ('Rewrite the failed experiment now. In sample_weights return a positive vector even '
                      'for negative standardized targets. Use the quantile-weight expression shown in the '
                      'execution error if it matches your intent, or choose a different valid experiment. '
                      'Return the complete experiment JSON.')
        correction = dict(revision_request=diagnostic, required_next=action)
        messages.extend([{'role': 'assistant', 'content': feedback['previous_reply']},
                         {'role': 'user', 'content': json.dumps(correction, ensure_ascii=False)}])
    write(folder/'messages.json', messages)
    max_tokens = min(rnd.max_output_tokens, MAX_OUTPUT_TOKENS)
    print(json.dumps(dict(dataset=rnd.dataset_key, round_index=rnd.round_index, stage=name,
                          host_max_output_tokens=rnd.max_output_tokens, max_tokens=max_tokens,
                          enable_thinking=False)),
          file=sys.stderr, flush=True)
    started = time.monotonic()
    receipt = rnd.model(messages, temperature=0, max_tokens=max_tokens, enable_thinking=False)
    write(folder/'receipt.json', dict(call_id=receipt.call_id, response=receipt.response,
                                     gateway_wall_seconds=time.monotonic()-started))
    return receipt


def decision(rnd, root, name, prompt, view, evaluate):
    """A confirmed unusable reply can be revised before any host-side commit."""
    feedback = None
    for attempt in range(1, 4):
        label = name if attempt == 1 else f'{name}-{attempt}'
        receipt = call_model(rnd, root, label, prompt, view, feedback)
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
                            instruction=('Correct the reported error and return one complete new decision. '
                                         'For an invalid training proposal change the executable expression that failed. '
                                         'Changing its explanation alone repeats the same error.'))
            write(folder/'rejection.json', feedback)
            if attempt == 3:
                raise


def selection(text, parent_id, candidate_id):
    value = json_value(text)
    if not isinstance(value, dict) or set(value) != {'selected_role', 'reason'}:
        raise ValueError('Return exactly selected_role ("parent" or "candidate") and reason.')
    role = value['selected_role']
    if role not in ('parent', 'candidate'):
        raise ValueError('selected_role must be the literal "parent" or "candidate".')
    reason = value['reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode('utf-8')) > 4000:
        raise ValueError('Selection reason must be nonempty and <=4000 bytes')
    return dict(selected_role=role, selected_artifact_id=parent_id if role == 'parent' else candidate_id, reason=reason)


def previous_decision_prompt(prompt, view, previous, condition):
    """A retained parent's last experiment is comparable only on the same measured inputs."""
    if not isinstance(previous, dict):
        return prompt
    previous_parent = previous.get('supplied_parent_artifact_id')
    if not previous_parent or previous.get('selected_artifact_id') != previous_parent:
        return prompt
    old = previous.get('condition') or {'dataset_key': previous.get('dataset_key')}
    changed = [key for key, value in condition.items() if old.get(key) and value and old[key] != value]
    unknown = [key for key, value in condition.items() if not old.get(key) or not value]
    if previous_parent != view['parent_artifact_id']:
        changed.append('parent_artifact_id')
    if not changed and not unknown:
        return prompt + ("\nThe previous experiment was rejected on THIS parent. Read its exact program, "
                         "training settings and measurement in previous_native_state. Repeating the "
                         "same executable settings on this seed will repeat the result. In reflection "
                         "name that experiment and identify a materially different change; implement it.")
    view['previous_measurement_scope'] = dict(
        previous_round=previous.get('round_index'), previous_condition=old, current_condition=condition,
        changed_fields=changed, unverified_fields=unknown,
        previous_parent_artifact_id=previous_parent,
        current_parent_artifact_id=view['parent_artifact_id'])
    return prompt + ("\nThe previous decision retained the supplied parent under the conditions recorded "
                     "in previous_measurement_scope. Its candidate measurement applies to that previous "
                     "experiment. Use parent_training_metrics and parent_validation_metrics measured on "
                     "this round's inputs. Test your proposal under the current conditions.")


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
        # The full SDK history stays in the audit record. Both decisions see every
        # round's measured result, while only the latest experiment needs its code.
        trend = []
        for item in history:
            point = {k: item[k] for k in ('id', 'round_index', 'stage_id', 'selected_artifact_id',
                     'validation_metrics', 'hypothesis') if k in item}
            plan = (item.get('configuration') or {}).get('plan')
            if plan is not None:
                point['plan'] = plan
            trend.append(point)
        previous_experiment = ({k: previous[k] for k in ('proposal', 'reason', 'selected_artifact_id',
                               'candidate_receipt', 'supplied_parent_artifact_id') if k in previous}
                               if previous else None)
        view.update(round_index=rnd.round_index, actual_parent_file_sha256=hashes['parent'],
                    parent_checkpoint_receipt=rnd.parent.checkpoint_sha256,
                    parent_artifact_id=rnd.parent.artifact_id, task_contract=rnd.task,
                    actual_platform_history=trend, previous_native_state=previous_experiment,
                    required_next='experiment')
        if rnd.dataset_key == 'B':
            from .experiment_agent.matched_history import open_history, append_result, prompt_history
            history_file = state/'bearing-history.json'
            last_b = json_value(history_file.read_text()) if history_file.is_file() else None
            matched = open_history(last_b, rnd.dataset_key, rnd.stage_id, hashes)
            if matched['experiments']:
                view['matched_experiments'] = prompt_history(matched)
        if not history:
            from .round_prompt import START_PROGRAMS
            view['baseline_program'] = START_PROGRAMS[rnd.dataset_key]
            if rnd.dataset_key == 'A':
                view['starting_options'] = {'steady_channel_dropout': False}
            view['reference_experiment'] = 'Test this parent-only starting program; explain the hypothesis and falsifier. Use the observed measurements to guide the next proposal.'
        stage = 'proposal_model'
        prompt = TASK_PROPOSAL_PROMPTS[rnd.dataset_key]
        condition = dict(dataset_key=rnd.dataset_key, stage_id=rnd.stage_id, **hashes)
        prompt = previous_decision_prompt(prompt, view, previous, condition)
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
                  (model.meta['steps'], report['core_steps_after'], report['model_kind'], report['program_sha256']))
        configuration = dict(program_sha256=report['program_sha256'],
                             plan=report['numerical']['plan'],
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
        select_view = dict(allowed_artifact_ids=allowed, allowed_roles=['parent', 'candidate'], parent_artifact=rnd.parent.receipt,
                           candidate_receipt=submitted, task_contract=rnd.task,
                           actual_platform_history=trend, proposal={k: proposal[k] for k in
                               ('hypothesis', 'expected_result', 'falsifier') if k in proposal},
                           executed_plan=report['numerical']['plan'],
                           local_diagnostics={'scope':'not platform measurements',
                               'before':report['local_validation_before'],
                               'after':report['local_validation_after']})
        if rnd.dataset_key == 'B' and matched['experiments']:
            select_view['matched_experiments'] = prompt_history(matched, report)
        if rnd.dataset_key == 'A':
            select_view['same_round_macro_f1'] = {label: report['local_validation_' + phase]['macro_f1']
                                                  for label, phase in [('parent', 'before'), ('candidate', 'after')]}
        if rnd.dataset_key == 'C':
            before = report['local_validation_before']
            after = report['local_validation_after']
            tail = 'log_mean_exp_asymmetric_error'
            select_view['retention_test'] = {
                'scope': 'Matched local validation rows; lower is better for both metrics.',
                'candidate_rmse_no_higher': after['rmse'] <= before['rmse'],
                'candidate_NASA_penalty_no_higher': after[tail] <= before[tail],
                'candidate_minus_parent_log_mean_penalty': after[tail] - before[tail],
                'parent_id': allowed[0],
                'candidate_id': allowed[1],
                'decision_rule': 'Protect the gains already measured. Adopt only for a Pareto improvement: both tests must be true and at least one metric must improve. If either test is false, retain the parent. Compare the actual paired evidence and explain the choice.',
            }
        def choose(folder, text):
            try:
                chosen = selection(text, rnd.parent.artifact_id, submitted['artifact_id'])
            except (TypeError, ValueError, RecursionError) as exc:
                raise RejectedProposal(str(exc)) from exc
            write(folder/'action.json', chosen)
            return chosen

        selection_prompt = SELECT_PROMPT + '\n' + TASK_SELECTION_HINTS[rnd.dataset_key]
        choice_receipt, chosen = decision(rnd, root, 'selection', selection_prompt, select_view, choose)
        stage = 'state_save'
        write(state/'last-round.json', dict(dataset_key=rnd.dataset_key, round_index=rnd.round_index,
              condition=condition, supplied_parent_artifact_id=rnd.parent.artifact_id, candidate_receipt=submitted,
              selected_artifact_id=chosen['selected_artifact_id'], reason=chosen['reason'],
              proposal=action, selection_receipt={'call_id':choice_receipt.call_id, 'response':choice_receipt.response},
              model_call_ids=list(rnd.call_ids), history_ids=[item['id'] for item in history]))
        if rnd.dataset_key == 'B':
            append_result(matched, rnd.round_index, proposal['program'], report, chosen['selected_role'], chosen['reason'])
            write(history_file, matched)
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
def prediction_model(raw, model_dir):
    # The official prediction process owns one immutable artifact mount.
    path, directory = Path(raw), Path(model_dir)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise HarnessError('ARTIFACT_OUTSIDE_MODEL_MOUNT')
    current = path
    while current != directory and current != current.parent:
        if current.is_symlink():
            raise HarnessError('ARTIFACT_SYMLINK')
        current = current.parent
    return Model.load(path)


def predict(sample, model_dir):
    raw = os.environ.get('CONTEST_ARTIFACT_PATH')
    if not raw:
        raise HarnessError('ARTIFACT_PATH_REQUIRED')
    model = prediction_model(raw, model_dir)
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
