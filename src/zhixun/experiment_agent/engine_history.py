"""Compare two official MLPs on equal-weight samples of previously released training projections."""
from pathlib import Path
import numpy as np


def compare_seen_training(state, train_digest, current_x, current_y, parent, candidate, round_index=0):
    if len(train_digest)!=64 or any(c not in '0123456789abcdef' for c in train_digest):
        raise ValueError('Expected a full SHA-256 training digest')
    history = Path(state) / 'engine-training'
    history.mkdir(exist_ok=True)
    path = history / (train_digest + '.npz')
    if not path.exists():
        sample = np.random.default_rng(42).choice(len(current_y), min(len(current_y),128), replace=False)
        remaining = np.setdiff1d(np.arange(len(current_y)), sample)
        fit = np.random.default_rng(43).choice(remaining, min(128, len(remaining)), replace=False)
        sample.sort()
        np.savez(path, x=current_x[sample], y=current_y[sample],
                 fit_x=current_x[fit], first_round=round_index)
    files = sorted(history.glob('*.npz'))
    errors = []
    for filename in files:
        with np.load(filename,allow_pickle=False) as rows:
            x,targets = rows['x'],rows['y']
        errors.append([float(np.mean((model.predict(x)-targets)**2)) for model in (parent,candidate)])
    before,after = np.mean(errors,axis=0)
    return dict(projections=len(files), rows_per_projection_limit=128,
                parent_mse=float(before),candidate_mse=float(after),
                passes=bool(len(files)==1 or after<=before))


def previous_training_inputs(state, train_digest):
    blocks = []
    for path in (Path(state) / 'engine-training').glob('*.npz'):
        if path.stem != train_digest:
            with np.load(path, allow_pickle=False) as data:
                if 'fit_x' in data and len(data['fit_x']):
                    blocks.append((int(data['first_round']), path.stem, data['fit_x']))
    return [rows for _, _, rows in sorted(blocks, key=lambda item: item[:2])]


def follows_history_conflict(previous, train_digest, parent_digest, round_index):
    if not previous or previous.get('dataset_key') != 'C':
        return False
    condition = previous.get('condition', {})
    return (previous.get('round_index') == round_index - 1
            and previous.get('engine_history_conflict') is True
            and condition.get('train') == train_digest
            and condition.get('parent') == parent_digest)
