"""Local UCI batch protocol. This is not the organizer's undisclosed evaluation SDK."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .engine import Config, Engine


PROTOCOL = 'gas-whole-batch-v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_batch(path, seed, batch):
    X, y = load_svmlight_file(str(path), n_features=128, zero_based=False)
    X = X.toarray()
    if not np.isfinite(X).all() or not np.isin(y, np.arange(1, 7)).all():
        raise ValueError('Invalid gas features or class IDs')
    # The source is class-grouped; file row order is not a timestamp.
    # Shuffle only WITHIN a batch, independent of its labels; never cross batches.
    order = np.random.default_rng(np.random.SeedSequence([seed, batch])).permutation(len(y))
    return X[order], y[order].astype(int), order + 1


def initial_split(y, seed):
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=0.4, stratify=y, random_state=seed)
    tune, gate = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return dict(train_indices=train, tune_indices=tune, gate_indices=gate)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def run(data, output, seed=42, use_experience=True, policy='rule', estimator='rf', max_iter=6, train_seconds=2.0, windows=(256, 768), include_followup=True):
    data, output = Path(data).resolve(), Path(output).resolve()
    config = Config(seed=seed, use_experience=use_experience, include_followup=include_followup, policy=policy,
                    estimator=estimator, max_iter=max_iter, train_seconds=train_seconds, windows=tuple(windows))
    files = [data / f'batch{i}.dat' for i in range(1, 11)]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError('All ten UCI batch files are required')
    engine = Engine(config, output)
    started = time.monotonic()
    # Freeze provenance and policy before any training or evaluation.
    write_json(output / 'protocol.json', dict(
        protocol=PROTOCOL, config=asdict(config), source='UCI Gas Sensor Array Drift Dataset',
        data_sha256={p.name: digest(p) for p in files},
        code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        bootstrap='batch1 only; stratified 60/20/20 train/tune/gate',
        evaluation='batches2..10 in source batch order; predict whole batch before labels',
        within_batch_order='seeded label-independent permutation; not physical time',
        adaptation='at most one attempt after nonterminal batch feedback; final labels score and resolve probation without further training or driver calls',
        metrics='sample-weighted error and equal-batch mean error; not official scores',
        limitation='public development data, local CPU baseline; no claim about private B or LLM',
        python=platform.python_version(), platform=platform.platform(),
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'flaml', 'river')}))
    X, y, source_rows = read_batch(files[0], seed, 1)
    split = initial_split(y, seed)
    write_json(output / 'bootstrap_split.json', {
        k: dict(source_rows=source_rows[v].tolist(),
                class_counts={str(c): int(np.sum(y[v] == c)) for c in range(1, 7)})
        for k, v in split.items()})
    engine.bootstrap(X, y, **split)
    audit_rows = [{'offset': i, 'batch': 1, 'source_row': int(row)}
                  for i, row in enumerate(source_rows)]
    batch_results, all_y, all_predictions, all_static = [], [], [], []
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir()
    for batch, path in enumerate(files[1:], 2):
        X, y, source_rows = read_batch(path, seed, batch)
        offset = engine.observed
        version = engine.version
        prediction = engine.predict_batch(offset, X)
        static = engine.pending['static_prediction'].copy()
        # Persist predictions without labels BEFORE passing labels into the engine.
        committed = prediction_dir / f'batch{batch}.csv'
        np.savetxt(committed, np.column_stack([source_rows, prediction, static]),
                   delimiter=',', fmt='%d', header='source_row,adaptive,static', comments='')
        engine._event('prediction_artifact_committed', batch=batch, sha256=digest(committed))
        before = engine.attempts
        engine.observe_labels(offset, y, final=batch == len(files))
        audit_rows.extend({'offset': offset + i, 'batch': batch, 'source_row': int(row)}
                          for i, row in enumerate(source_rows))
        result = dict(batch=batch, rows=len(y), predicted_version=version,
                      adaptive_error=float(1 - accuracy_score(y, prediction)),
                      static_error=float(1 - accuracy_score(y, static)),
                      adaptive_macro_f1=float(f1_score(y, prediction, average='macro', zero_division=0)),
                      static_macro_f1=float(f1_score(y, static, average='macro', zero_division=0)),
                      adaptive_confusion=confusion_matrix(y, prediction, labels=range(1, 7)).tolist(),
                      attempted_after_batch=engine.attempts - before)
        batch_results.append(result)
        all_y.extend(y.tolist()); all_predictions.extend(prediction.tolist()); all_static.extend(static.tolist())
        write_json(output / 'batch_results.json', batch_results)
        print(json.dumps({k: v for k, v in result.items() if k != 'adaptive_confusion'}), flush=True)
    write_json(output / 'row_audit.json', audit_rows)
    summary = dict(engine.summary(), protocol=PROTOCOL, config=asdict(config),
                   equal_batch_adaptive_error=float(np.mean([r['adaptive_error'] for r in batch_results])),
                   equal_batch_static_error=float(np.mean([r['static_error'] for r in batch_results])),
                   adaptive_macro_f1=float(f1_score(all_y, all_predictions, average='macro')),
                   static_macro_f1=float(f1_score(all_y, all_static, average='macro')),
                   wall_seconds=time.monotonic() - started,
                   final_probation_unresolved=engine.probation is not None,
                   llm_calls=engine.decisions, official_evaluation=False)
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    summary['training_wall_seconds'] = sum(e.get('training', {}).get('wall_seconds', 0) for e in events)
    write_json(output / 'gas_result.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Frozen local whole-batch UCI gas evaluation')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--windows', nargs='+', type=int, default=[256, 768])
    parser.add_argument('--estimator', choices=['rf', 'extra_tree', 'lgbm'], default='rf')
    parser.add_argument('--max-iter', type=int, default=6)
    parser.add_argument('--train-seconds', type=float, default=2.0)
    parser.add_argument('--policy', choices=['rule', 'pytrio'], default='rule')
    parser.add_argument('--without-experience', action='store_true')
    parser.add_argument('--without-followup', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output, args.seed, not args.without_experience, args.policy,
                         args.estimator, args.max_iter, args.train_seconds, args.windows,
                         not args.without_followup), indent=2))


if __name__ == '__main__':
    main()
