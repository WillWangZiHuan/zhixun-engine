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


from .official import digest, score, improves


def run(bundle, output, *, selection, rounds=10, timeout=120, driver_python=None, price_record=None, training_runtime, epochs_per_candidate=1, batch_size=None, initial_learning_rate=None):
    if selection not in ('joint_model', 'joint_grid', 'balanced_grid', 'balanced_reduce_lr') or type(rounds) is not int or not 1 <= rounds <= 10:
        raise ValueError('explicit selection mode and 1..10 rounds required')
    if type(epochs_per_candidate) is not int or not 1 <= epochs_per_candidate <= 10000:
        raise ValueError('epochs_per_candidate must be in 1..10000')
    if batch_size is not None and (type(batch_size) is not int or not 1 <= batch_size <= 65536):
        raise ValueError('batch_size must be in 1..65536')
    if initial_learning_rate is not None and (type(initial_learning_rate) not in (int, float) or not math.isfinite(initial_learning_rate) or not 0 < initial_learning_rate <= .5):
        raise ValueError('initial_learning_rate must be finite and in (0,.5]')
    if selection in ('joint_model', 'joint_grid') and (epochs_per_candidate != 1 or batch_size is not None or initial_learning_rate is not None):
        raise ValueError('joint decision protocol uses its original one-epoch and checkpoint hyperparameters')
    training_runtime = Path(training_runtime).resolve()
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite subprocess timeout required')
    runtime_files = [training_runtime/'baselines'/f for f in ('train.py','model.py','data.py','predict.py')]
    runtime_hashes = {str(p.relative_to(training_runtime)): digest(p) for p in runtime_files}
    prices = None
    if selection == 'joint_model':
        from datetime import datetime, timezone
        from .joint_driver import uncached_estimate
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
    if profile not in ('B', 'C'):
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
    runtime_config = training_runtime/f'baselines/configs/{profile}.json'
    if json.loads(runtime_config.read_text()) != config:
        raise ValueError('runtime configuration differs from the official task bundle')
    runtime_hashes[str(runtime_config.relative_to(training_runtime))] = digest(runtime_config)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'protocol': 'official-public-continuation-v2', 'profile': profile, 'task': task,
                'selection': selection, 'group_weight': 'selected_per_round', 'rounds': rounds, 'epochs_per_candidate': epochs_per_candidate, 'batch_size': batch_size if batch_size is not None else config['batch_size'], 'initial_learning_rate': base_rate,
                'hyperparameters': selection, 'same_public_data_every_round': True,
                'selection_data_is_evaluation_data': True, 'independent_generalization_test': False,
                'official_platform_protocol': False, 'source': str(bundle), 'source_hashes': expected,
                'runner_sha256': digest(__file__), 'python': sys.version, 'training_runtime_hashes': runtime_hashes,
                'subprocess_timeout_seconds': timeout, 'model_api_call_limit': 10 if selection == 'joint_model' else 0, 'platform_evaluations': 0}
    if selection == 'joint_model':
        from . import joint_driver as official_driver, joint_worker as official_worker
        from zhixun import policy_worker
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

    train_rows = [json.loads(line) for line in (bundle/'public/train.jsonl').read_text().splitlines()]
    group_key = 'bearing' if profile == 'B' else 'source_subset'
    training_group_counts = dict(Counter(row['meta'][group_key] for row in train_rows))
    def group_metrics(prediction_file):
        pred = [json.loads(line)['outputs']['value'] for line in prediction_file.read_text().splitlines()]
        errors = {}
        for row, value in zip(rows,pred): errors.setdefault(row['meta'][group_key],[]).append(value-row['answer'])
        return {g:{'n':len(e),'rmse':math.sqrt(sum(v*v for v in e)/len(e)),'mae':sum(abs(v) for v in e)/len(e),'bias':sum(e)/len(e)} for g,e in sorted(errors.items())}
    selected = bundle/'checkpoint.npz' 
    selected_metrics = evaluate(selected, 'initial')
    for key, value in selected_metrics.items():
        if not math.isclose(value, baseline['validation_after'][key], abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError('initial predictions disagree with official report')
    initial_metrics = selected_metrics.copy()
    initial_groups = group_metrics(output/"initial-predictions.jsonl")
    initial_checkpoint = digest(selected)
    initial_steps = baseline['final_steps']
    selected_steps = initial_steps
    events = []
    api_usage = []
    with (output/'events.jsonl').open('w') as log, (output/'decisions.jsonl').open('w') as decisions:
        for n in range(1, rounds+1):
            parent_hash = digest(selected)
            candidate_dir = output/f'round-{n:02d}'
            started = time.monotonic()
            decision = {'round': n, 'based_on_rounds': list(range(n)), 'parent_sha256': parent_hash, 'epochs': epochs_per_candidate, 'batch_size': batch_size if batch_size is not None else config['batch_size']}
            if selection in ('balanced_grid', 'balanced_reduce_lr'):
                if selection == 'balanced_grid':
                    tried = {e['learning_rate'] for e in events if e['parent_sha256'] == parent_hash}
                    learning_rate = next(base_rate * 2**k for k in [0, 1, *range(-1, -14, -1)] if base_rate * 2**k not in tried)
                else:
                    from .official import propose_learning_rate
                    learning_rate = propose_learning_rate(base_rate, events)
                group_weight = 'balanced'
                decision.update(learning_rate=learning_rate, loss_mode=group_weight, driver='history_rule_no_llm', rule=selection)
            else:
                from .joint_driver import build_request, decide, parse_decision
                request = build_request(profile, base_rate, parent_hash, events, seed=baseline['seed'], initial_metrics=initial_metrics, initial_groups=initial_groups, training_group_counts=training_group_counts, initial_checkpoint=initial_checkpoint)
                if selection == 'joint_model':
                    proposal, usage = decide(output/f'api-{n:02d}', request, python=driver_python, prices=prices)
                    api_usage.append(usage)
                else:
                    proposal = parse_decision(json.dumps({'action_id':request['candidates'][0]['action_id'],'hypothesis':'Test first untried joint action on the current parent.','evidence_rounds':[n-1]}),request)
                learning_rate, group_weight = proposal['learning_rate'], proposal['loss_mode']
                decision.update(action_id=proposal['action_id'], learning_rate=learning_rate, loss_mode=group_weight, driver='designated_model' if selection=='joint_model' else 'joint_grid', hypothesis=proposal['hypothesis'], evidence_rounds=proposal['evidence_rounds'], rule='joint action; fixed overall RMSE gate')
                (output/f'request-{n:02d}.json').write_text(json.dumps(request,indent=2)+'\n')
            decisions.write(json.dumps(decision)+'\n'); decisions.flush()
            command = [sys.executable, '-B', '-m', 'baselines.train', '--profile', profile,
                       '--train', 'public/train.jsonl', '--validation', 'public/validation.jsonl',
                       '--from-baseline' if n == 1 else '--resume', str(selected), '--epochs', str(epochs_per_candidate),
                       '--output', str(candidate_dir), '--stage-id', f'{selection}-{n:02d}']
            command.extend(['--learning-rate', str(learning_rate)])
            command += ['--group-weight', group_weight]
            if batch_size is not None:
                command += ['--batch-size', str(batch_size)]
            command[command.index('--train')+1] = str(bundle/'public/train.jsonl')
            command[command.index('--validation')+1] = str(bundle/'public/validation.jsonl')
            result = subprocess.run(command, cwd=training_runtime, env=env, capture_output=True, text=True, timeout=timeout)
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
            if report['loss_mode'] != group_weight: raise ValueError('executed loss mismatch')
            candidate_groups = group_metrics(output/f'candidate-{n:02d}-predictions.jsonl')
            accepted = improves(metrics, selected_metrics, task)
            before = selected_metrics.copy()
            if accepted:
                selected, selected_metrics, selected_steps = candidate, metrics, report['final_steps']
            training_history = [json.loads(line) for line in (candidate_dir/'history.jsonl').read_text().splitlines()]
            event = {'round': n, 'epochs': epochs_per_candidate, 'batch_size': report['batch_size'], 'loss_mode': group_weight, 'candidate_groups': candidate_groups, 'driver': decision['driver'], 'learning_rate': learning_rate, 'parent_sha256': parent_hash,
                     'training_loss': training_history[-1]['loss'],
                     'candidate_sha256': candidate_hash, 'candidate_metrics': metrics,
                     'incumbent_metrics_before': before, 'accepted': accepted,
                     'reason': 'strict primary-metric improvement' if accepted else 'no strict primary-metric improvement',
                     'selected_sha256': digest(selected), 'selected_metrics': selected_metrics,
                     'initial_steps': report['initial_steps'], 'candidate_final_steps': report['final_steps'],
                     'selected_steps': selected_steps, 'elapsed_seconds': time.monotonic()-started}
            log.write(json.dumps(event, ensure_ascii=False)+'\n');log.flush()
            events.append(event)
            print(json.dumps({'profile': profile, **event}), flush=True)
    for path, value in expected.items():
        if digest(bundle/path) != value:
            raise ValueError(f'original source changed: {path}')
    for path, value in runtime_hashes.items():
        if digest(training_runtime/path) != value:
            raise ValueError(f'training runtime changed: {path}')
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
    p.add_argument('--selection', choices=['joint_grid', 'joint_model', 'balanced_grid', 'balanced_reduce_lr'], required=True)
    p.add_argument('--training-runtime', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--epochs-per-candidate', type=int, default=1)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--initial-learning-rate', type=float)
    p.add_argument('--driver-python', type=Path)
    p.add_argument('--price-record', type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.bundle, args.output, selection=args.selection, rounds=args.rounds, training_runtime=args.training_runtime, timeout=args.timeout, driver_python=args.driver_python, price_record=args.price_record, epochs_per_candidate=args.epochs_per_candidate, batch_size=args.batch_size, initial_learning_rate=args.initial_learning_rate)))


if __name__ == '__main__':
    main()
