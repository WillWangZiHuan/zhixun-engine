import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from .engine import Config, Engine


def synthetic(task, seed, rows, concept):
    """No concept ID is exposed to the engine; only numeric features and delayed labels."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 6))
    sign = 1 if concept % 2 == 0 else -1
    signal = sign * np.where(X[:, 0] >= 0, 1.0, -1.0)
    y = (signal > 0).astype(int) if task == 'classification' else 3 * signal + rng.normal(0, 0.15, rows)
    return X, y


def demo(args):
    config = Config(task=args.task, seed=args.seed)
    engine = Engine(config, args.output)
    X, y = synthetic(args.task, args.seed, 1024, 0)
    engine.bootstrap(X, y, train_indices=np.arange(768),
                     tune_indices=np.arange(768, 896), gate_indices=np.arange(896, 1024))
    boundaries = []
    # Includes stationary periods and repeated abrupt changes of the target relation.
    for segment, concept in enumerate([0, 1, 0, 1, 0]):
        boundaries.append(dict(start=engine.observed, concept=concept))
        X, y = synthetic(args.task, args.seed + segment + 1, 1536, concept)
        for start in range(0, len(y), 128):
            batch_id = engine.observed
            engine.predict_batch(batch_id, X[start:start + 128])
            # Synthetic evaluator reveals labels only AFTER the prediction was committed.
            engine.observe_labels(batch_id, y[start:start + 128])
        print(json.dumps({'segment': segment, **engine.summary()}, ensure_ascii=False), flush=True)
    summary = engine.summary()
    summary.update(config=asdict(config), synthetic_boundaries=boundaries,
                   platform=platform.platform(), python=platform.python_version(),
                   versions={p: importlib.metadata.version(p) for p in ('river', 'flaml', 'numpy', 'scikit-learn')})
    (Path(args.output) / 'demo_result.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Result: {Path(args.output).resolve() / 'demo_result.json'}")


def main():
    parser = argparse.ArgumentParser(description='驭变队 · 智训引擎本地闭环')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('demo', help='Run a synthetic drift stream with a frozen baseline')
    run.add_argument('--task', choices=['classification', 'regression'], default='classification')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--output', type=Path, required=True, help='New directory; existing results are never overwritten')
    show = commands.add_parser('inspect', help='Read a completed or interrupted run')
    show.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'demo':
        demo(args)
    else:
        print((args.output / 'summary.json').read_text())


if __name__ == '__main__':
    main()
