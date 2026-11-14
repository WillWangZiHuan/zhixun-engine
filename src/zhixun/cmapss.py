"""Group-disjoint C-MAPSS static baseline with untouched official endpoint targets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split

from .engine import Config
from .training import train_candidate


def read_table(path):
    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] != 26 or not np.isfinite(table).all():
        raise ValueError('C-MAPSS requires 26 finite columns')
    if not np.equal(table[:, :2], np.floor(table[:, :2])).all():
        raise ValueError('Unit and cycle IDs must be integers')
    units = np.unique(table[:, 0]).astype(int)
    if not np.array_equal(units, np.arange(1, len(units) + 1)):
        raise ValueError('Unexpected source unit IDs')
    for unit in units:
        cycles = table[table[:, 0] == unit, 1]
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            raise ValueError('Unsorted, repeated or missing unit cycles')
    return table


def train_targets(table):
    targets = np.empty(len(table))
    for unit in np.unique(table[:, 0]):
        mask = table[:, 0] == unit
        targets[mask] = table[mask, 1].max() - table[mask, 1]
    return targets


def endpoint_rows(table):
    return np.array([np.flatnonzero(table[:, 0] == unit)[-1] for unit in np.unique(table[:, 0])])


def group_split(table, seed):
    units = np.unique(table[:, 0]).astype(int)
    train, rest = train_test_split(units, test_size=.4, random_state=seed)
    tune, gate = train_test_split(rest, test_size=.5, random_state=seed)
    return {'train': train, 'tune': tune, 'gate': gate}


def evaluate(data, output, subset, seed=42):
    data, output = Path(data).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_file, test_file, target_file = [data / f'{prefix}_{subset}.txt' for prefix in ('train', 'test', 'RUL')]
    train = read_table(train_file)
    y = train_targets(train)
    X = train[:, 1:]  # Observed cycle + 3 settings + 21 sensors; never unit identity.
    groups = group_split(train, seed)
    masks = {name: np.isin(train[:, 0], values) for name, values in groups.items()}
    config = asdict(Config(task='regression', seed=seed, train_seconds=3, max_iter=24))
    protocol = dict(protocol='cmapss-static-group-v1', subset=subset, seed=seed,
                    train_file_sha256=hashlib.sha256(train_file.read_bytes()).hexdigest(),
                    test_file_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest(),
                    unit_groups={k: v.tolist() for k, v in groups.items()},
                    features=['cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)],
                    label='Uncapped train-unit maximum observed cycle minus current cycle.',
                    selection='Minimum tuning MAE over held-out units; no gate or official test targets for selection.',
                    gate='All-cycle MAE on separate train-source units; not official endpoint metric.',
                    test='One endpoint prediction per official test engine; official endpoint RUL loaded afterward.',
                    adaptation=False, official_competition_evaluation=False, config=config)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    started, candidates = time.monotonic(), []
    for estimator in ('rf', 'extra_tree', 'lgbm'):
        model, info = train_candidate(output / 'experiments' / estimator,
            X[masks['train']], y[masks['train']], X[masks['tune']], y[masks['tune']],
            config | {'estimator': estimator})
        candidates.append((info['tuning_loss'], estimator, model, info))
    _, chosen, model, info = min(candidates, key=lambda c: c[0])
    gate_mae = float(np.abs(model.predict(X[masks['gate']]) - y[masks['gate']]).mean())
    test = read_table(test_file)
    endpoints = endpoint_rows(test)
    prediction = np.asarray(model.predict(test[endpoints, 1:]))
    # Prevent negative lifetime predictions by a predeclared physical lower bound.
    prediction = np.maximum(prediction, 0)
    np.savetxt(output / 'test_predictions.csv', np.column_stack([test[endpoints, 0], prediction]),
               delimiter=',', header='unit,predicted_rul', comments='')
    truth = np.loadtxt(target_file)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError('Official RUL file must contain one nonnegative target per test engine')
    result = dict(subset=subset, selected_estimator=chosen, tuning_mae=info['tuning_loss'],
                  group_gate_all_cycle_mae=gate_mae, test_engines=len(truth),
                  test_endpoint_mae=float(np.abs(prediction - truth).mean()),
                  test_endpoint_rmse=float(np.sqrt(np.mean((prediction - truth) ** 2))),
                  wall_seconds=time.monotonic() - started,
                  candidates={name: metadata for _, name, _, metadata in candidates},
                  target_sha256=hashlib.sha256(target_file.read_bytes()).hexdigest(),
                  adaptation=False, scope='public static development baseline, not competition score')
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='C-MAPSS public static endpoint baseline')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
        evaluate(args.data, args.output / subset, subset, args.seed)


if __name__ == '__main__':
    main()
