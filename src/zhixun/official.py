"""Local CPU continuation runner for the supplied contest bundles.

This is a development protocol, not the unpublished platform scheduling SDK.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(rows, predictions, task, classes):
    if not rows or len(rows) != len(predictions):
        raise ValueError('prediction count must equal the nonempty validation set')
    if task == 'classification':
        predicted = []
        for response in predictions:
            outputs = response['outputs']
            if len(outputs) != len(classes) or [o['label'] for o in outputs] != classes:
                raise ValueError('prediction class order does not match checkpoint configuration')
            probs = [o['score'] for o in outputs]
            if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs):
                raise ValueError('invalid class probability')
            if abs(sum(probs)-1) > 1e-8:
                raise ValueError('class probabilities must sum to one')
            predicted.append(classes[max(range(len(classes)), key=lambda i: probs[i])])
        truth = [r['answer'] for r in rows]
        if any(t not in classes for t in truth):
            raise ValueError('unknown validation class')
        tp = Counter(t for t, p in zip(truth, predicted) if t == p)
        actual, guessed = Counter(truth), Counter(predicted)
        observed = [c for c in classes if c in actual or c in guessed]
        return {'macro_f1': sum(2*tp[c]/(actual[c]+guessed[c]) for c in observed)/len(observed),
                'accuracy': sum(t == p for t, p in zip(truth, predicted))/len(truth)}
    error = []
    for row, response in zip(rows, predictions):
        value = response['outputs']['value']
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('regression predictions must be finite and nonnegative')
        error.append(value-row['answer'])
    return {'rmse': math.sqrt(sum(e*e for e in error)/len(error)),
            'mae': sum(abs(e) for e in error)/len(error)}


def improves(candidate, incumbent, task):
    return candidate['macro_f1'] > incumbent['macro_f1'] if task == 'classification' else candidate['rmse'] < incumbent['rmse']


def propose_learning_rate(initial, history):
    if not history:
        return initial
    last = history[-1]
    return last['learning_rate'] if last['accepted'] else last['learning_rate'] / 2


def run(bundle, output, *, selection, rounds=10, timeout=120, driver_python=None, price_record=None, epochs_per_candidate=1, batch_size=None, initial_learning_rate=None):
    if selection not in ('always', 'validation', 'reduce_lr', 'pytrio') or type(rounds) is not int or not 1 <= rounds <= 10:
        raise ValueError('explicit selection mode and 1..10 rounds required')
    if type(epochs_per_candidate) is not int or not 1 <= epochs_per_candidate <= 10000:
        raise ValueError("epochs_per_candidate must be in 1..10000")
    if batch_size is not None and (type(batch_size) is not int or not 1 <= batch_size <= 65536):
        raise ValueError('batch_size must be in 1..65536')
    if initial_learning_rate is not None and (type(initial_learning_rate) not in (int,float) or not math.isfinite(initial_learning_rate) or not 0 < initial_learning_rate <= 1):
        raise ValueError('initial_learning_rate must be finite and in (0,1]')
    if selection == "pytrio" and (epochs_per_candidate != 1 or batch_size is not None or initial_learning_rate is not None):
        raise ValueError("the validated model decision protocol specifies one epoch and checkpoint hyperparameters")
    prices = None
    if selection == 'pytrio':
        from datetime import datetime, timezone
        from .official_driver import uncached_estimate
        if rounds != 10 or not driver_python or not price_record:
            raise ValueError('model comparison requires ten rounds, a driver Python and current price record')
        driver_python = Path(driver_python).absolute()
        if not driver_python.is_file() or not os.environ.get('PYTRIO_API_KEY'):
            raise ValueError('designated-model interpreter and authentication are required')
        prices = json.loads(Path(price_record).read_text())
        checked = datetime.fromisoformat(prices['checked_at'])
        if checked.tzinfo is None or checked.astimezone(timezone.utc).date() != datetime.now(timezone.utc).date():
            raise ValueError('refresh the account quote before this comparison')
        uncached_estimate(4096, 256, prices)
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError('results must be outside the unchanged source bundle')
    baseline = json.loads((bundle/'training.json').read_text())
    info = json.loads((bundle/'baseline-info.json').read_text())
    profile = info['dataset_key']
    if profile not in ('A', 'B', 'C'):
        raise ValueError('unknown task profile')
    config = json.loads((bundle/f'baselines/configs/{profile}.json').read_text())
    if config['task'] != ('classification' if profile == 'A' else 'regression'):
        raise ValueError('use the task-specific current bundle, not a historical B profile')
    task = config['task']
    base_rate = baseline['learning_rate'] if initial_learning_rate is None else initial_learning_rate
    expected = {'checkpoint.npz': baseline['checkpoint_sha256'],
                'public/train.jsonl': baseline['train_sha256'],
                'public/validation.jsonl': baseline['validation_sha256'],
                f'baselines/configs/{profile}.json': baseline['profile_sha256']}
    expected.update({f'baselines/{k}': v for k, v in baseline['source_sha256'].items()})
    for path, value in expected.items():
        if digest(bundle/path) != value:
            raise ValueError(f'source integrity mismatch: {path}')
    if info['checkpoint_sha256'] != expected['checkpoint.npz']:
        raise ValueError('baseline metadata conflict')
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'protocol': 'official-public-continuation-v2', 'profile': profile, 'task': task,
                'selection': selection, 'rounds': rounds, 'epochs_per_candidate': epochs_per_candidate, 'batch_size': batch_size if batch_size is not None else config['batch_size'], 'initial_learning_rate': base_rate,
                'hyperparameters': 'bounded designated-model learning-rate proposals' if selection == 'pytrio' else 'halve last proposed learning rate after rejection' if selection == 'reduce_lr' else 'checkpoint defaults', 'same_public_data_every_round': True,
                'selection_data_is_evaluation_data': True, 'independent_generalization_test': False,
                'official_platform_protocol': False, 'source': str(bundle), 'source_hashes': expected,
                'runner_sha256': digest(__file__), 'python': sys.version,
                'subprocess_timeout_seconds': timeout, 'model_api_call_limit': 9 if selection == 'pytrio' else 0, 'platform_evaluations': 0}
    if selection == 'pytrio':
        from . import official_driver, official_worker, policy_worker
        protocol['driver_sources'] = {Path(m.__file__).name: digest(m.__file__) for m in (official_driver, official_worker, policy_worker)}
        protocol['price_record'] = prices
        protocol['driver_python'] = str(driver_python)
    (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    rows = [json.loads(s) for s in (bundle/'public/validation.jsonl').read_text().splitlines() if s.strip()]
    request_file = output/'prediction-inputs.jsonl'
    with request_file.open('w') as f:
        for row in rows:
            f.write(json.dumps({'task': task, 'inputs': row['inputs']})+'\n')
    env = {k:v for k,v in os.environ.items() if k not in ('PYTRIO_API_KEY', 'SWANLAB_API_KEY')}
    env.update(BASELINE_CPU_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def evaluate(checkpoint, label):
        path = output/(label+'-predictions.jsonl')
        with request_file.open() as inputs, path.open('w') as predictions:
            result = subprocess.run([sys.executable, '-B', '-m', 'baselines.predict', '--checkpoint', str(checkpoint)],
                                    cwd=bundle, env=env, stdin=inputs, stdout=predictions,
                                    stderr=subprocess.PIPE, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr)
        predictions = [json.loads(s) for s in path.read_text().splitlines()]
        return score(rows, predictions, task, config.get('classes', []))

    selected = bundle/'checkpoint.npz'
    selected_metrics = evaluate(selected, 'initial')
    for key, value in selected_metrics.items():
        if not math.isclose(value, baseline['validation_after'][key], abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError('initial predictions disagree with official report')
    initial_metrics = selected_metrics.copy()
    initial_steps = baseline['final_steps']
    selected_steps = initial_steps
    events = []
    api_usage = []
    with (output/'events.jsonl').open('w') as log, (output/'decisions.jsonl').open('w') as decisions:
        for n in range(1, rounds+1):
            parent_hash = digest(selected)
            candidate_dir = output/f'round-{n:02d}'
            started = time.monotonic()
            learning_rate = propose_learning_rate(base_rate, events) if selection == 'reduce_lr' else base_rate
            decision = {'round': n, 'based_on_rounds': list(range(1, n)), 'parent_sha256': parent_hash,
                        'learning_rate': learning_rate, 'epochs': epochs_per_candidate, 'batch_size': batch_size if batch_size is not None else config['batch_size'],
                        'rule': 'halve after previous rejection; keep after acceptance' if selection == 'reduce_lr' else 'fixed checkpoint defaults',
                        'driver': 'history_rule_no_llm' if selection == 'reduce_lr' else 'fixed_rule_no_llm'}
            if selection == 'pytrio' and n > 1:
                from .official_driver import build_request, decide
                request = build_request(profile, baseline['learning_rate'], parent_hash, events, seed=baseline['seed'])
                proposal, usage = decide(output/f'api-{n:02d}', request, python=driver_python, prices=prices)
                api_usage.append(usage)
                learning_rate = proposal['learning_rate']
                decision.update(learning_rate=learning_rate, driver='designated_model',
                                rule='bounded model proposal with independent fixed validation gate',
                                hypothesis=proposal['hypothesis'], evidence_rounds=proposal['evidence_rounds'])
            decisions.write(json.dumps(decision)+'\n'); decisions.flush()
            command = [sys.executable, '-B', '-m', 'baselines.train', '--profile', profile,
                       '--train', 'public/train.jsonl', '--validation', 'public/validation.jsonl',
                       '--from-baseline' if n == 1 else '--resume', str(selected), '--epochs', str(epochs_per_candidate),
                       '--output', str(candidate_dir), '--stage-id', f'{selection}-{n:02d}']
            if selection in ('reduce_lr', 'pytrio') or initial_learning_rate is not None:
                command.extend(['--learning-rate', str(learning_rate)])
            if batch_size is not None:
                command.extend(['--batch-size', str(batch_size)])
            result = subprocess.run(command, cwd=bundle, env=env, capture_output=True, text=True, timeout=timeout)
            if result.returncode:
                raise RuntimeError(result.stderr)
            report = json.loads((candidate_dir/'training.json').read_text())
            candidate = candidate_dir/'checkpoint.npz'
            candidate_hash = digest(candidate)
            if report['parent_checkpoint_sha256'] != parent_hash or digest(selected) != parent_hash:
                raise ValueError('candidate parent chain mismatch')
            if report['initial_steps'] != selected_steps or report['final_steps'] <= selected_steps:
                raise ValueError('optimizer step continuity mismatch')
            if candidate_hash != report['checkpoint_sha256']:
                raise ValueError('candidate checkpoint digest mismatch')
            metrics = evaluate(candidate, f'candidate-{n:02d}')
            for key, value in metrics.items():
                if not math.isclose(value, report['validation_after'][key], abs_tol=1e-8, rel_tol=1e-10):
                    raise ValueError('independent prediction metrics disagree with training report')
            accepted = selection == 'always' or improves(metrics, selected_metrics, task)
            before = selected_metrics.copy()
            if accepted:
                selected, selected_metrics, selected_steps = candidate, metrics, report['final_steps']
            training_history = [json.loads(line) for line in (candidate_dir/'history.jsonl').read_text().splitlines()]
            event = {'round': n, 'driver': decision['driver'], 'learning_rate': learning_rate, 'parent_sha256': parent_hash,
                     'training_loss': training_history[-1]['loss'],
                     'candidate_sha256': candidate_hash, 'candidate_metrics': metrics,
                     'incumbent_metrics_before': before, 'accepted': accepted,
                     'reason': 'always continue' if selection == 'always' else 'strict primary-metric improvement' if accepted else 'no strict primary-metric improvement',
                     'selected_sha256': digest(selected), 'selected_metrics': selected_metrics,
                     'initial_steps': report['initial_steps'], 'candidate_final_steps': report['final_steps'],
                     'selected_steps': selected_steps, 'elapsed_seconds': time.monotonic()-started}
            log.write(json.dumps(event, ensure_ascii=False)+'\n');log.flush()
            events.append(event)
            print(json.dumps({'profile': profile, **event}), flush=True)
    for path, value in expected.items():
        if digest(bundle/path) != value:
            raise ValueError(f'original source changed: {path}')
    shutil.copyfile(selected, output/'selected-checkpoint.npz')
    summary = {'status': 'completed', 'initial_metrics': initial_metrics, 'final_metrics': selected_metrics,
               'rounds': len(events), 'accepted': sum(e['accepted'] for e in events),
               'training_epochs_executed': len(events) * epochs_per_candidate,
               'optimizer_steps_executed': sum(e['candidate_final_steps']-e['initial_steps'] for e in events),
               'unique_candidate_checkpoints': len({e['candidate_sha256'] for e in events}),
               'selected_checkpoint_sha256': digest(output/'selected-checkpoint.npz'),
               'source_unchanged': True, 'official_score': False, 'model_api_calls': len(api_usage),
               'estimated_uncached_cny': sum(u['estimated_uncached_cny'] for u in api_usage),
               'actual_bill': False, 'platform_evaluations': 0}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection', choices=['always', 'validation', 'reduce_lr', 'pytrio'], required=True)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--epochs-per-candidate', type=int, default=1)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--initial-learning-rate', type=float)
    p.add_argument('--driver-python', type=Path)
    p.add_argument('--price-record', type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.bundle, args.output, selection=args.selection, rounds=args.rounds, driver_python=args.driver_python, price_record=args.price_record, epochs_per_candidate=args.epochs_per_candidate, batch_size=args.batch_size, initial_learning_rate=args.initial_learning_rate)))


if __name__ == '__main__':
    main()
