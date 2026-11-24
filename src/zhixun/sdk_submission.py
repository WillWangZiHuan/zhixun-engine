"""Workspace-v2 protocol: propose, continue, measure, select, persist, acknowledge."""
from pathlib import Path
from functools import lru_cache
import ast
import hashlib
import time
import json
import math
import os
import re
import sys
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
                {'role':'user', 'content':json.dumps(view, ensure_ascii=False, allow_nan=False)}]
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
    receipt = rnd.model(messages, temperature=0, max_tokens=max_tokens, enable_thinking=False)
    write(folder/'receipt.json', dict(call_id=receipt.call_id, response=receipt.response))
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


def experiment_key(proposal, mode, parent, training, validation, stage):
    # Whitespace and comments do not turn an identical program into a new experiment.
    source = proposal.get('program')
    if not isinstance(source, str):
        raise ValueError('proposal.program must be ONE JSON STRING with all five def statements and newline '
                         'separators. A JSON map of function names is invalid. Follow last_executed_program or '
                         'baseline_program for the structure. plan(context) returns a dictionary with exactly '
                         'epochs and batch_size. Put batch_limit in proposal, outside plan. Return a complete corrected experiment.')
    program = ast.dump(ast.parse(source), include_attributes=False)
    values = [mode, proposal.get('batch_limit'), program, parent, training, validation, stage]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


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
        probes = None
        adaptive = rnd.dataset_key in ('B', 'C')
        if adaptive:
            from .experiment_agent.probes import paired_probes
            from .sdk_training import metrics
            pilot_wall = time.monotonic()
            pilot_cpu = time.process_time()
            ledger = previous.get('experiments', []) if isinstance(previous, dict) else []
            last = ledger[-1] if ledger else None
            tail = 'log_mean_exp_asymmetric_error'
            tail_conflict = (rnd.dataset_key == 'C' and last is not None and
                             last['parent'] == rnd.parent.artifact_id and last['selected'] == 'parent' and
                             last['after']['rmse'] < last['before']['rmse'] and last['after'][tail] > last['before'][tail])
            probes, measured = paired_probes(model, prepared[1], prepared[2], metrics, tail_conflict)
            probe_cost = {'wall_seconds': time.monotonic() - pilot_wall,
                          'cpu_seconds': time.process_time() - pilot_cpu}
            view['paired_probes'] = measured
            if history and previous and isinstance(previous.get('proposal'), dict):
                last_program = previous['proposal'].get('proposal', {}).get('program')
                if isinstance(last_program, str):
                    view['last_executed_program'] = last_program
                    view.pop('baseline_program', None)
            view['budget_example'] = ('batch_limit=1 runs a single real Adam update after the trained pilot. '
                                      'Omit the limit for all batches. The prior ledger reports actual optimizer '
                                      'updates and metrics on each side of Adam.')
            view['stage_id'] = rnd.stage_id
            view['actual_platform_history'] = [
                {key: row[key] for key in ('id','round_index','selected_artifact_id','validation_metrics') if key in row}
                for row in history]
            view['previous_native_state'] = [
                {k: v for k, v in entry.items() if k != 'signature'} for entry in ledger]
        if not history:
            from .round_prompt import START_PROGRAMS
            view['baseline_program'] = START_PROGRAMS[rnd.dataset_key]
            if rnd.dataset_key == 'A':
                view['starting_options'] = {'steady_channel_dropout': False}
            view['reference_experiment'] = 'Test this parent-only starting program; explain the hypothesis and falsifier. Use the observed measurements to guide the next proposal.'
        stage = 'proposal_model'
        prompt = TASK_PROPOSAL_PROMPTS[rnd.dataset_key]
        if adaptive:
            prompt += ('\nYou have matched train-only pilot updates on the actual parent. paired_probes contains '
                       'their scores on the current public validation rows BEFORE Adam. Set proposal.update_mode '
                       'to the name of the pilot to continue. reference uses the usual task update; adam_only '
                       'starts at the parent. long_life_head updates only the existing readout with long-life '
                       'weighting; tail_conditioned uses tail-sensitive conditioned gradients; _half takes half '
                       'the alternative displacement. All then execute your authored Adam program. Base this '
                       'choice on current measured numbers, including the competing metric on C. Preserve a '
                       'promising pilot with a gentle Adam control. Use the whole ledger of paired trials '
                       'to avoid repeating an unsuccessful combination on the same parent and data. State '
                       'one falsifiable prediction relative to this parent. Keep explanations brief.')
        retained_last_parent = (isinstance(previous, dict) and previous.get('supplied_parent_artifact_id') == rnd.parent.artifact_id
                                and previous.get('selected_artifact_id') == rnd.parent.artifact_id)
        gas_recovery = (rnd.dataset_key == 'A' and rnd.round_index > 1 and isinstance(previous, dict)
                        and previous.get('selected_artifact_id') == rnd.parent.artifact_id
                        and (retained_last_parent or previous.get('gas_controls_enabled') is True))
        if gas_recovery:
            from .round_prompt import START_PROGRAMS
            controls = {}
            for mode in ('reference', 'adam_only'):
                folder = root/('gas-control-' + mode)
                folder.mkdir()
                control = dict(operation='core_code', hypothesis='Measure the gas auxiliary update on the current parent.',
                               expected_result='Different validation Macro-F1.', falsifier='Equal validation Macro-F1.',
                               program=START_PROGRAMS['A'], update_mode=mode)
                _, measured = continue_candidate(folder, prepared, control, gas_recovery=True)
                scores = measured['local_validation_after']
                controls[mode] = dict(macro_f1=scores['macro_f1'], accuracy=scores['accuracy'],
                                      adam_steps=measured['core_steps_after'] - measured['core_steps_before'],
                                      checkpoint_sha256=measured['checkpoint_sha256'])
            view['gas_controls'] = dict(program=START_PROGRAMS['A'], steady_channel_dropout=False,
                                        parent_sha256=hashes['parent'], train_sha256=hashes['train'],
                                        validation_sha256=hashes['validation'], results=controls)
            prompt += (' After the rollback, gas_controls contains two completed experiments on THIS parent and current '
                       'validation rows using the same reference Adam program. reference includes the auxiliary gas '
                       'update; adam_only omits it and inherits the parent Adam moments. Compare their Macro-F1 and '
                       'the parent score. Choose proposal.update_mode as reference or adam_only. You may reuse the '
                       'measured program and dropout setting or propose a different test. The resulting candidate '
                       'will be measured before you decide whether to adopt it.')
        if gas_recovery and retained_last_parent:
            prompt += ("\nThe previous experiment was rejected on THIS parent. Read its exact program, "
                       "training settings and measurement in previous_native_state. Repeating the "
                       "same executable settings on this seed will repeat the result. In reflection "
                       "name that experiment and identify a materially different change; implement it.")
        if adaptive:
            prompt += ('\nSet optional proposal.batch_limit (integer 1..256) to control the TOTAL Adam updates, '
                       'or omit it for full epochs. For a promising measured pilot, first use a one-update '
                       'control with a small learning rate so its measured change can be attributed. After seeing '
                       'the pilot-to-final difference in the ledger, choose whether to lengthen or redirect '
                       'the next experiment. C balanced interpolates the two train-only update directions. '
                       'Use the actual RMSE and tail numbers, not the order of the modes. '
                       'First compare the small control against this stage, then design the next test. '
                       'Use the supplied source string as a syntax example. proposal.program must be a single JSON '
                       'string defining the five functions. plan(context) returns exactly {"epochs":1,"batch_size":1024} '
                       'with your chosen values. batch_limit belongs beside program, outside the Python plan. '
                       'Limit reflection to three sentences and the selection reason to two.')
        if adaptive:
            prompt += (' A repeated, previously rejected executable experiment on the very same parent, split '
                       'and stage is a step-size comparison: the harness halves the displacement from the parent '
                       'and the effective Adam learning rate at each repetition. Changed data or parent starts '
                       'a fresh comparison. You may instead design another experiment. On B compare the short-life '
                       'and long-life public errors as well as the overall metric. residual_head is a damped readout update '
                       'weighted by training residual size. Pick a trade-off justified by the measured table.')
        if rnd.dataset_key == 'C' and tail_conflict:
            prompt += (' The previous attempt reduced RMSE but increased the NASA penalty on this parent. '
                       'Your new measured options calibrated_reference and calibrated_half make a training-only '
                       'asymmetric-loss output-bias adjustment after a full or half reference step. '
                       'Check which options improve both current metrics versus adam_only. If a trade-off persists, '
                       'design a numerical experiment and state which measurement would justify accepting it.')
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
            if adaptive:
                proposed = action['proposal']
                try:
                    sig = experiment_key(proposed, proposed.get('update_mode', 'reference'),
                                         hashes['parent'], hashes['train'], hashes['validation'], rnd.stage_id)
                except (KeyError, SyntaxError, TypeError, ValueError, RecursionError) as exc:
                    raise RejectedProposal(str(exc)) from exc
                repeats = sum(entry.get('signature') == sig and entry['selected'] == 'parent' for entry in ledger)
            else:
                repeats = 0
            write(folder/'action.json', action)
            candidate, report = continue_candidate(folder, prepared, action['proposal'], probes, 0.5 ** repeats, gas_recovery=gas_recovery)
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
        if gas_recovery:
            configuration.update(update_mode=report['update_mode'], gas_controls=controls,
                                 adam_steps=report['core_steps_after'] - report['core_steps_before'])
            change += ' Measured both gas update modes after rollback; continued with ' + report['update_mode'] + '.'
        if adaptive:
            configuration.update(update_mode=report['update_mode'], trial_scale=report['trial_scale'], batch_limit=proposal.get('batch_limit'),
                                 training_start_sha256=report['training_start_sha256'],
                                 candidate_sha256=report['checkpoint_sha256'], pilot_cost=probe_cost)
            change = (f"From supplied parent, tested {len(probes)} train-only updates; continued "
                      f"{report['update_mode']} at scale {report['trial_scale']:.4g} for {report['core_steps_after'] - report['core_steps_before']} Adam steps. "
                      f"Paired RMSE {report['local_validation_before']['rmse']:.5g} to "
                      f"{report['local_validation_after']['rmse']:.5g}.")
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
                           actual_platform_history=history, proposal=action,
                           local_diagnostics={'scope':'not platform measurements',
                               'before':report['local_validation_before'],
                               'after':report['local_validation_after']})
        if adaptive:
            select_view.pop('actual_platform_history')
            select_view['stage_id'] = rnd.stage_id
            select_view['trial_scale'] = report['trial_scale']
            select_view['parent_artifact'] = {'artifact_id': rnd.parent.artifact_id}
            select_view['proposal'] = {'hypothesis': proposal['hypothesis'], 'update_mode': report['update_mode']}
            select_view['matched_pair'] = select_view.pop('local_diagnostics')
            select_view['instruction'] = ('Use only measurements paired on current validation rows. '
                                          'An earlier receipt for the same parent can describe another stage. '
                                          'Give a concise reason with the two measured values.')
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
        if adaptive:
            ledger = ledger + [dict(round=rnd.round_index, stage=rnd.stage_id,
                          parent=rnd.parent.artifact_id, validation_sha256=hashes['validation'],
                          experiment=proposal['hypothesis'][:320], mode=report['update_mode'],
                          plan=configuration['plan'], learning_rate=report['numerical']['history'][0]['learning_rate_first'],
                          batch_limit=proposal.get('batch_limit'), steps=report['core_steps_after']-report['core_steps_before'],
                          before_adam=report['before_adam'], trial_scale=report['trial_scale'],
                          signature=experiment_key(proposal, report['update_mode'], hashes['parent'], hashes['train'],
                                                   hashes['validation'], rnd.stage_id),
                          before=report['local_validation_before'], after=report['local_validation_after'],
                          selected=chosen['selected_role'])]
        write(state/'last-round.json', dict(**({'experiments': ledger} if adaptive else {}), **({'gas_controls_enabled': True} if gas_recovery else {}), dataset_key=rnd.dataset_key, round_index=rnd.round_index,
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
    return Model.load(Path(path))


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
