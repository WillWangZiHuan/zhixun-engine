"""Compare two official MLPs on equal-weight samples of previously released training projections."""
from pathlib import Path
import numpy as np


def compare_seen_training(state, train_digest, current_x, current_y, parent, candidate):
    if len(train_digest)!=64 or any(c not in '0123456789abcdef' for c in train_digest):
        raise ValueError('Expected a full SHA-256 training digest')
    history = Path(state) / 'engine-training'
    history.mkdir(exist_ok=True)
    path = history / (train_digest + '.npz')
    if not path.exists():
        sample = np.random.default_rng(42).choice(len(current_y), min(len(current_y),128), replace=False)
        sample.sort()
        np.savez(path,x=current_x[sample],y=current_y[sample])
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
